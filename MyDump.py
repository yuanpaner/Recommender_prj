""" use pickle to save algorithm and data to avoid repeated calculation
"""
import pickle
import os
from MovieLens import MovieLens

folder_path = './DumpFiles'

if not os.path.exists(folder_path):
    print(f"{folder_path} doesn't exist. trying to make")
    os.makedirs(folder_path)

def Save(file_name, predictions=None, algo=None, data = None, verbose = 0):
    save_file_name = folder_path+'/' + file_name
    if os.path.exists(save_file_name):
        print(f'{save_file_name} file exist, dump save DOESN"T proceed')
    else:
        # _, loaded_algo = dump.load(file_name)
        dump_obj = {'predictions': predictions,
                    'algo': algo,
                    'data': data
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
            print(f"{load_file_name} LOAD successfully")
        return dump_obj['predictions'], dump_obj['algo'], dump_obj['data']
    else:
        if verbose:
            print("file doesn't exist, loading FAILS")
        return None, None, None


def LoadMovieLensData(loader = False):
    if loader:
        _, _, data = Load('ratingsDataset',1)
        _, _, rankings = Load('rankings',1)
        _, _, ml = Load('ml',1)
        # ml = MovieLens()
        if data == None or rankings == None or ml == None:
            ml = MovieLens()
            print("\nLoading movie ratings...")
            data = ml.loadMovieLensLatestSmall() # will assign ml the name-id dictionary
            print("Computing movie popularity ranks so we can measure novelty later...")
            rankings = ml.getPopularityRanks() # for novelty
            Save('ratingsDataset', data = data, verbose = 1)
            Save('rankings', data = rankings, verbose = 1)
            Save('ml', data = ml, verbose = 1)
    else :
        ml = MovieLens()
        print("\nNo loader, Loading movie ratings...")
        data = ml.loadMovieLensLatestSmall()
        print("No loader, Computing movie popularity ranks so we can measure novelty later...")
        rankings = ml.getPopularityRanks()
    # print("\nLoading movie ratings...")
    # data = ml.loadMovieLensLatestSmall()
    # print("Computing movie popularity ranks so we can measure novelty later...")
    # rankings = ml.getPopularityRanks() # for novelty
    return (ml, data, rankings)
