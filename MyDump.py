""" use pickle to save algorithm and data to avoid repeated calculation
"""
import pickle
import os

folder_path = './DumpFiles'

if not os.path.exists(folder_path):
    print(f"{folder_path} doesn't exist. trying to make")
    os.makedirs(folder_path)

def Save(file_name, predictions=None, algo=None, verbose = 0):
    save_file_name = folder_path+'/' + file_name
    if os.path.exists(save_file_name):
        print(f'{save_file_name} file exist, saving doesn"t proceed')
    else:
        # _, loaded_algo = dump.load(file_name)
        dump_obj = {'predictions': predictions,
                    'algo': algo
                    }
        pickle.dump(dump_obj, open(save_file_name, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        # if algo is not None:
        #     dump_obj['algo'] = algo.__dict__  # add algo attributes
        #     dump_obj['algo']['name'] = algo.__class__.__name__
        #     pickle.dump(dump_obj, open(save_file_name, 'wb'))
        #     print(f'{save_file_name} SAVE sucessfully')
        # else:
        #     print('Please provide the algorithm')
        if verbose:
            print(f'The dump has been saved as file {save_file_name}')

def Load(file_name, verbose = 0):
    load_file_name = folder_path+'/' + file_name
    if os.path.exists(load_file_name):
        dump_obj = pickle.load(open(load_file_name, 'rb'))
        if verbose:
            print(f"{load_file_name} load successfully")
        return dump_obj['predictions'], dump_obj['algo']
    else:
        if verbose:
            print("file doesn't exist, loading fails")
        return None, None

def SaveData(file_name, data, verbose = 0):
    save_file_name = folder_path+'/' + file_name
    dump_obj = {'data': data}
    pickle.dump(dump_obj, open(save_file_name, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f'The dump has been saved as file {save_file_name}')

def LoadData(file_name, verbose = 0):
    load_file_name = folder_path+'/' + file_name
    if os.path.exists(load_file_name):
        dump_obj = pickle.load(open(load_file_name, 'rb'))
        if verbose:
            print(f"{load_file_name} load successfully")
        return dump_obj['data']
    else:
        if verbose:
            print(f"{load_file_name} doesn't exist, loading fails")
        return None