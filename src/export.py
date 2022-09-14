from os import XATTR_CREATE
import torch
import onnx
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
ckp = utility.Checkpoint(args)


def main():
    if ckp.ok:
        loader = data.Data(args)
        _model = model.Model(args, ckp)
        print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
        # 查看输入
        loader_train = loader.loader_train
        loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(loader_train):
            print(batch)
            print(lr.shape)
            print(hr.shape)
            break

        _model.get_model().eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 3, 48, 48, device=device)

        torch.onnx.export(_model.get_model(), dummy_input, 'NLSN.onnx', verbose=True, opset_version=12)

        ckp.done()


# # Example X2 SR
# !python main.py --dir_data ../ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4
# --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1
# --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save_dir /content/drive/MyDrive/sr_output/NLSN
# --save NLSN_x2 --data_train DIV2K --load NLSN_x2
if __name__ == '__main__':
    main()
