import segmentation_models_pytorch as smp
from tests.test_segmentation.unet_seg_dataset import UnetSegDataset
from torch.utils.data import DataLoader
from medcam import medcam

if __name__ == '__main__':
    dataset = UnetSegDataset("cuda")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = smp.Unet(classes=1)
    model.cuda()
    model = medcam.inject(model, backend="gcam", output_dir="'results/smp/gcam'", save_maps=True)

    model.eval()
    for batch in data_loader:
        img = batch["img"]
        print("Test")
        _ = model(img)
