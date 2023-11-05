from src.train import  test_model
import json
import time
import os
from multiprocessing import Pool

def multi_run_wrapper(args):
   return test_model(*args)


os.makedirs('results', exist_ok=True)

if __name__ == "__main__":

    start = time.time()
    with open('models/model_params.json', 'r') as infile:
        model_params = json.load(infile)

    source_folder = "data/benchmarks"
    folder_paths = [ "vdata"]

    folder_number = 0 
    for folder_path in folder_paths:
        save_results = []
        filenames = []
        for file_name in os.listdir(source_folder + folder_path):
            if os.path.isfile(os.path.join(source_folder + folder_path, file_name)):
                filenames.append(file_name)

        for f in filenames:
            models = []
            for v in model_params:
                models.append((v["name"], source_folder + folder_path, f))

            # List of parameters
            with Pool(len(models)) as p:
                res = p.map(multi_run_wrapper, models)

            result =  min([r[0] for r in res])
            start_time = min([r[1] for r in res])
            end_time = max([r[2] for r in res])
            print(f, result, end_time - start_time)
            save_results.append({"score": result, "name": f, "time": end_time - start_time})

        with open('results/results_'+ folder_path+'.json', 'w') as outfile:
            json.dump({"results": save_results}, outfile)
    folder_number+=1