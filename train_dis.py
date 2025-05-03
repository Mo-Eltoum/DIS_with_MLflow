from utils import *
import torch.optim as optim
from data_loader import *
import argparse
import os


def main(restore_model, start_ite, model_digit, input_size, early_stop, model_save_fre,
         batch_size_train, batch_size_valid, max_ite, max_epoch_num, pretrained_path, data_aug, lr, multi_gpu):

    # --- Step 0: Build datasets ---
    # check data availability
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        print('ERROR: Data not available')
        return

    # get train and val directories
    train_dir = os.path.join(os.path.join(os.getcwd(), "data/train"))
    val_dir = os.path.join(os.path.join(os.getcwd(), "data/val"))
    train_datasets, valid_datasets = create_inputs(train_dir, val_dir)

    # --- Step 1: Build dataloaders ---
    print("--- create training dataloader ---")
    # collect training dataset
    train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    # build dataloader for training datasets
    if data_aug == 1:
        train_tf = [
            GOSResize(input_size),
            GOSRandomHFlip(),
            GOSRandomVFlip(),
            GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ]
    else:
        train_tf = [
            GOSResize(input_size),
            GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        ]
    val_tf = [
        GOSResize(input_size),
        GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ]
    train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                           my_transforms=train_tf,
                                                           batch_size=batch_size_train,
                                                           shuffle=True)
    print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    # build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    # build dataloader for training datasets
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                           my_transforms=val_tf,
                                                           batch_size=batch_size_valid,
                                                           shuffle=False)

    print(len(valid_dataloaders), " valid dataloaders created")

    # create dataset json file
    # train dataset
    train_info = dataset_info(os.path.join(train_dir, 'im'), os.path.join(train_dir, 'gt'))
    save_json_file(train_info, os.path.join(train_dir, 'train.json'))
    # val dataset
    train_info = dataset_info(os.path.join(val_dir, 'im'), os.path.join(val_dir, 'gt'))
    save_json_file(train_info, os.path.join(val_dir, 'val.json'))

    # --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")

    # initiate the model
    net = ISNetDIS()

    # apply transfer learning and load pre-trained weights
    if pretrained_path != "":
        pretrained_dict = torch.load(pretrained_path)
        # Load pre-trained weights into the model
        model_dict = net.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict:
                model_dict[k] = v
        net.load_state_dict(model_dict)

    # convert to half precision
    if model_digit == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if restore_model != "":
        print("restore model from:")
        model_path = os.path.join(os.getcwd(), 'weights')
        print(model_path + "/" + restore_model)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path + "/" + restore_model))
        else:
            net.load_state_dict(torch.load(model_path + "/" + restore_model, map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # --- Step 4: applying Distributed Data Parallelism ---
    if multi_gpu:
        net = nn.DataParallel(net)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

    # --- Step 3: Setup Mlflow tracking ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("DIS")

    # --- Step 4: Train and Valid Model with MLFlow logs ---

    # Start a run with the next available run name
    current_run_number = mlflow.get_experiment_by_name("DIS").tags.get("current_run_number", "0")
    current_run_number = int(current_run_number)

    # Determine the next available run name
    next_run_name = f"dis-{current_run_number}"
    mlflow.set_experiment("DIS")

    with mlflow.start_run(run_name=next_run_name) as run:
        # Set the run name tag
        mlflow.set_tag("mlflow.runName", next_run_name)
        # Log hyperparameters
        mlflow.log_params({
            "max_epochs": max_epoch_num,
            "train_batch_size": batch_size_train,
            "val_batch_size": batch_size_valid,
            "max_iterations": max_ite,
            "early_stop": early_stop,
            "model_save_frequency": model_save_fre,
            "learning_rate": lr,
        })
        # Increment the run number for the next run
        current_run_number += 1
        mlflow.set_experiment("DIS")
        mlflow.set_experiment_tags({"current_run_number": str(current_run_number)})

        train(net,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              start_ite, model_digit, early_stop, model_save_fre, batch_size_train,
              batch_size_valid, max_ite, max_epoch_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_gpu', default=False)
    parser.add_argument('--restore_model', default="")
    parser.add_argument('--start_ite', default=0)
    parser.add_argument('--model_digit', default="full")
    parser.add_argument('--input_size', default=[1024, 1024])
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--model_save_fre', type=int, default=1000)
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_valid', type=int, default=1)
    parser.add_argument('--max_ite', type=int, default=1000000)
    parser.add_argument('--max_epoch_num', type=int, default=100000)
    parser.add_argument('--pretrained_weights', type=str, default="")
    parser.add_argument('--data_aug', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    main(args.restore_model, args.multi_gpu, args.start_ite, args.model_digit, args.input_size,args.early_stop,
         args.model_save_fre, args.batch_size_train, args.batch_size_valid, args.max_ite,
         args.max_epoch_num, args.pretrained_weights, args.data_aug, args.lr)
