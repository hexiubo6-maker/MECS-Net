import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import mmformer as mm
from utils.lr_scheduler import MultiEpochsDataLoader
from data.datasets_nii import Brats_loadall_test_nii


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_stage_features(model, x, mask):
    """
    Run a forward-like pass to collect features at 3 stages:
    - enc_x5: raw encoder x5 features per modality (stacked)
    - intra_x5: after IntraFormer per modality (stacked)
    - inter_x5: after InterFormer fused features

    Returns dict of tensors with shapes:
      enc_x5: [B, 5*C5, D5, H5, W5]
      intra_x5: [B, 5*Tdims, D5, H5, W5]
      inter_x5: [B, C5*5, D5, H5, W5]
    """
    model.eval()
    with torch.no_grad():
        # === Encoder === (copy from mmformer.Model.forward but stop where needed)
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = model.module.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = model.module.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = model.module.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = model.module.t2_encoder(x[:, 3:4, :, :, :])
        edge_x1, edge_x2, edge_x3, edge_x4, edge_x5 = model.module.edge_encoder(x[:, 4:5, :, :, :])

        enc_x5 = torch.cat((flair_x5, t1ce_x5, t1_x5, t2_x5, edge_x5), dim=1)

        # === IntraFormer on x5 ===
        transformer_basic_dims = mm.transformer_basic_dims
        patch_size = mm.patch_size
        flair_token_x5 = model.module.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1ce_token_x5 = model.module.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1_token_x5 = model.module.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t2_token_x5 = model.module.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        edge_token_x5 = model.module.edge_encode_conv(edge_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

        flair_intra_token_x5 = model.module.flair_transformer(flair_token_x5, model.module.flair_pos)
        t1ce_intra_token_x5 = model.module.t1ce_transformer(t1ce_token_x5, model.module.t1ce_pos)
        t1_intra_token_x5 = model.module.t1_transformer(t1_token_x5, model.module.t1_pos)
        t2_intra_token_x5 = model.module.t2_transformer(t2_token_x5, model.module.t2_pos)
        edge_intra_token_x5 = model.module.edge_transformer(edge_token_x5, model.module.edge_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        edge_intra_x5 = edge_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        x5_intra = model.module.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5, edge_intra_x5), dim=1), mask)

        intra_x5 = torch.cat((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5, edge_intra_x5), dim=1)

        # === InterFormer on x5 ===
        num_modals = mm.num_modals
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5, edge_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)
        multimodal_token_x5 = torch.cat((
            flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
            t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
            t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
            t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
            edge_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
        ), dim=1)
        multimodal_pos = torch.cat((model.module.flair_pos, model.module.t1ce_pos, model.module.t1_pos, model.module.t2_pos, model.module.edge_pos), dim=1)
        multimodal_inter_token_x5 = model.module.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        transformer_basic_dims_all = transformer_basic_dims * num_modals
        multimodal_inter_x5 = model.module.multimodal_decode_conv(
            multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), mm.patch_size, mm.patch_size, mm.patch_size, transformer_basic_dims_all)
                .permute(0, 4, 1, 2, 3).contiguous())
        inter_x5 = multimodal_inter_x5

        return {
            'enc_x5': enc_x5.detach(),
            'intra_x5': intra_x5.detach(),
            'inter_x5': inter_x5.detach(),
        }


def downsample_label_to(feat, label):
    """Nearest-neighbor downsample label to the spatial size of feat."""
    _, _, D, H, W = feat.shape
    label_ds = F.interpolate(label.float().unsqueeze(1), size=(D, H, W), mode='nearest').squeeze(1).long()
    return label_ds


def sample_features_by_label(feat, label, max_per_class=500):
    """
    feat: [B, C, D, H, W]
    label: [B, D, H, W] with int classes 0..K-1
    Return X [N, C], y [N]
    """
    B, C, D, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    label = label.view(-1)

    X_list, y_list = [], []
    classes = torch.unique(label)
    for cls in classes.tolist():
        idx = torch.nonzero(label == cls, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        if idx.numel() > max_per_class:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_per_class]
            idx = idx[perm]
        X_list.append(feat[idx])
        y_list.append(torch.full((idx.numel(),), cls, device=feat.device, dtype=torch.long))
    if not X_list:
        return None, None
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def tsne_and_plot(X, y, save_path, title):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    try:
        # 延迟导入，避免全局导入时因 SciPy/Sklearn 环境问题崩溃
        from sklearn.manifold import TSNE  # type: ignore
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        X2 = tsne.fit_transform(X)
        method = 't-SNE'
    except Exception as e:
        # 回退：使用 NumPy PCA 作为替代，让可视化不中断
        Xc = X - X.mean(axis=0, keepdims=True)
        # 使用 SVD 做 PCA
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            X2 = Xc @ Vt[:2].T
        except Exception:
            # 极端情况下退化到前两维（若已有>=2维）
            if X.shape[1] >= 2:
                X2 = X[:, :2]
            else:
                X2 = np.pad(X, ((0, 0), (0, 2 - X.shape[1])), mode='constant')
        method = 'PCA (fallback)'

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap='tab10', s=2, alpha=0.7)
    plt.title(f"{title} [{method}]")
    plt.legend(*scatter.legend_elements(), title='Class', loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='t-SNE feature visualization for mmformer components')
    parser.add_argument('--root', type=str, default='C:\\Users\\xiubo\\Desktop\\dataset', help='Dataset root directory')
    parser.add_argument('--resume', type=str,default='C:\\Users\\xiubo\\PycharmProjects\\hfn\\output\\model_last.pth', help='Path to model checkpoint .pth')
    parser.add_argument('--save_dir', type=str, default='C:\\Users\\xiubo\\PycharmProjects\\hfn\\tsne_plots', help='Directory to save plots')
    parser.add_argument('--max_per_class', type=int, default=500, help='Max samples per class')
    parser.add_argument('--num_samples', type=int, default=2, help='How many volumes to sample from the test set')
    parser.add_argument('--dataname', type=str, default='BRATS2020', choices=['BRATS2021', 'BRATS2020', 'BRATS2018', 'BRATS2015'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    test_set = Brats_loadall_test_nii(transforms='Compose([NumpyType((np.float32, np.int64)),])', root=args.root, test_file='')
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    num_cls = 4 if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018'] else 5

    model = mm.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    sampled = 0
    for i, data in enumerate(test_loader):
        x = data[0].cuda()    # [B, 5, D, H, W]
        target = data[1].cuda()  # [B, D, H, W]
        names = data[-1]
        # 规范化病例名，避免包含完整路径或非法字符导致保存失败
        sample_name_raw = str(names[0])
        sample_name = os.path.basename(sample_name_raw)
        sample_name = os.path.splitext(sample_name)[0]
        sample_name = sample_name.replace(':', '_').replace('\\', '_').replace('/', '_')
        if isinstance(data[2], torch.Tensor):
            mask = data[2]
        else:
            mask = torch.ones((x.size(0), 5), dtype=torch.bool)
        mask = mask.cuda()

        # 保证与模型 IntraFormer 位置编码长度一致：裁剪到 64x64x64 补丁
        # 输入 shape: [B, 5, D, H, W]
        D, H, W = x.size(2), x.size(3), x.size(4)
        in_patch = 64
        # 若原体素小于 64，则跳过该样本
        if D < in_patch or H < in_patch or W < in_patch:
            print(f"Skip {sample_name}: volume smaller than 64^3")
            continue
        d0 = (D - in_patch) // 2
        h0 = (H - in_patch) // 2
        w0 = (W - in_patch) // 2
        x_crop = x[:, :, d0:d0+in_patch, h0:h0+in_patch, w0:w0+in_patch]
        target_crop = target[:, d0:d0+in_patch, h0:h0+in_patch, w0:w0+in_patch]

        feats = collect_stage_features(model, x_crop, mask)

        # For each stage, downsample labels and sample features
        results = {}
        for stage, f in feats.items():
            lbl_ds = downsample_label_to(f, target_crop)
            X, y = sample_features_by_label(f, lbl_ds, max_per_class=args.max_per_class)
            if X is None:
                continue
            results[stage] = (X, y)

        # Plot
        for stage, (X, y) in results.items():
            save_path = os.path.join(args.save_dir, f"tsne_{stage}_{sample_name}.png")
            title = f"t-SNE {stage} - {sample_name}"
            tsne_and_plot(X, y, save_path, title)
            print(f"Saved: {save_path}")

        sampled += 1
        if sampled >= args.num_samples:
            break

    print('Done.')


if __name__ == '__main__':
    main()
