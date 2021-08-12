import os


args = {}

print("-------- Launch training for Autokart with the following pretrained model --------")
print("InriaAerial Pretrained Model              --> type 1 + Enter")
print("MappingChallenge Pretrained Model         --> type 2 + Enter")
print()
dataset = str(input("-->   "))
print()
#dataset = "1"
print("Launching training...")

if dataset == "1":
    args["config"] = "./configs/config.defaults.autokart_dataset_inria"
    args["run_name"] = "autokart_inria_model_512x512"
    args["init_run_name"] = "autokart_inria_model_512x512"
elif dataset == "2":
    args["config"] = "./configs/config.defaults.autokart_dataset_mapping"
    args["run_name"] = "autokart_mapping_model"
    args["init_run_name"] = "mapping_dataset.unet_resnet101_pretrained.train_val"


args["run_dir"] = "runs"
args["mode"] = "train"
args["fold"] = ["train"]
args["max_epoch"] = 245
args["batch_size"] = 2
args["gpus"] = 1



os.system("python main.py -c {} -r {} --run_name {} --init_run_name {} -m {} --fold {} -b {} --max_epoch {} -g {} ".format(args["config"], args["run_dir"], args["run_name"], args["init_run_name"], args["mode"], *args["fold"], args["batch_size"], args["max_epoch"], args["gpus"]))
