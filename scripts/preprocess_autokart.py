import os
import argparse
import tifffile

def get_args():
	argparser = argparse.ArgumentParser()

	argparser.add_argument(
		"-d", "--dataset",
		required = True,
		type = str,
		nargs = "*",
		choices = ['mapping', 'inria'],
		help = "Specify which dataset the pretrained model is trained on, determines how the images are preprocessed.")

	argparser.add_argument(
		"-f", "--fold",
		required = True,
		type = str,
		nargs = "*",
		help = "Specify which datafold to preprocess")


	args = argparser.parse_args()
	return args


def main():
	args = get_args()
	fold = args.fold[0]
	dataset = args.dataset[0]



	polygonize_args = {"filepath" : "/nr/samba/jodata10/pro/autokart/usr/maria/framefield/data/autokart_dataset/raw/{}/ground_truth/*.tif".format(fold),
			   "run_name" : "inria_dataset_osm_mask_only.unet16",
	           "runs_dirpath" : "runs",
			   "out_ext" : "geojson" if dataset == "inria" else "shp"}


	os.system("python ./../polygonize_mask.py -f {} -r {} --run_name {} --out_ext {}".format(polygonize_args["filepath"], polygonize_args["runs_dirpath"], polygonize_args["run_name"], polygonize_args["out_ext"]))

	if dataset == "mapping":

		shp_args = {"shp_dirpath" : "/nr/samba/jodata10/pro/autokart/usr/maria/framefield/data/autokart_dataset/raw/{}/ground_truth/shapefiles".format(fold),
			    "output_filepath" : "/nr/samba/jodata10/pro/autokart/usr/maria/framefield/data/autokart_dataset/raw/{}/annotation.json".format(fold)}

		os.system("python ./shp_to_json.py --shp_dirpath {} --output_filepath {}".format(shp_args["shp_dirpath"], shp_args["output_filepath"]))

main()
