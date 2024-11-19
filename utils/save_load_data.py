import pickle, json
import numpy as np

def save_pickle(*data, filename : str = 'filename'):
    '''save_pickle
    Save a new 'filename.pickle' file.
    '''        
    with open(filename + '.pickle', 'wb') as f: 
         pickle.dump(data,f) 
    return

def load_pickle(filename : str = 'filename'):
    '''load_pickle
    Load an existing 'filename.pickle' file and output its results.
    '''
    with open(filename + '.pickle', 'rb') as f: 
      return pickle.load(f)[0]  

     
def load_json(filename : str = 'filename', asDict=True): # Return as dictionary True or False
    if filename.split(".")[-1] != "json":
        filename += ".json"
    f = open(filename)

    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    # Convert lists to usable np.arrays
    data["cps"] = np.array(data["cps"])
    data["w"]   = np.array(data["w"])
    data["patchverts"] = np.array(data["patchverts"])
    data["patches"] = np.array(data["patches"]) 
    
    # Certain values had to be remapped because JSON did not support tuples as keys
    data["nelems"]   = remap_keys_reverse(data["nelems"])
    data["knotval"]  = remap_keys_reverse(data["knotval"])
    data["knotmult"] = remap_keys_reverse(data["knotmult"])

    if asDict:
        return data
    else:
        cps = data["cps"]
        w   = data["w"]
        patchverts = data["patchverts"]
        patches = data["patches"]
        nelems  = data["nelems"]
        knotval = data["knotval"]
        knotmult   = data["knotmult"]
        boundaries = data["boundaries"]
        return cps, w, patchverts, patches, nelems, knotval, knotmult, boundaries

def save_json(cps : np.ndarray, w : np.ndarray, patches : np.ndarray, patchverts : np.ndarray, 
              knotval : dict, knotmult: dict, nelems: dict, boundaries: dict, filename : str = 'filename'):
    # We first have to identify what is what (input ordering)
    # We only expect one or a combination of the following:
    # cps, w, patches, patchverts, boundaries, knotval, knotmult, nelems
    # for idata in data:
    #     if type(idata) == np.ndarray: # We either have, cps, w, patches, patchverts
    #         ...
    #     elif type(idata) == dict:
    #         ...
    #     else:
    #         raise TypeError(f"The provided input is of an unknown type {type(idata)}")
    if filename.split(".")[-1] != "json":
        filename += ".json"

    # Convert np.arrays to lists
    cps = cps.tolist()
    w   = w.tolist()
    patches    = patches.tolist()
    patchverts = patchverts.tolist()
    
    # Assign everything to one dictionary
    Data = {}
    Data["cps"] = cps
    Data["w"]   = w
    Data["patches"]    = patches
    Data["patchverts"] = patchverts
    Data["nelems"]     = remap_keys(nelems, convValfloat=True)
    Data["knotval"]    = remap_keys(knotval, convValfloat=True)
    Data["knotmult"]   = remap_keys(knotmult, convValfloat=True)
    Data["boundaries"] = boundaries
    json_object = json.dumps(Data, indent=4)

    with open(filename + ".json", "w") as outfile:
        outfile.write(json_object)
    return


def remap_keys(mapping, convValfloat=False): # Remapping keys reuired if tuple is a key in the initial dict
    # if convValfloat:
    return [{'key':[int(i) for i in k ], 'value': v} for k, v in mapping.items()]    
    # else:
    #    return [{'key':k, 'value': v} for k, v in mapping.items()]

def remap_keys_reverse(mapping, convListfloat=False, convValfloat=False):
    remapped = {}
    for map in mapping:
        k = map["key"]
        v = map["value"]
        remapped[tuple(k)] = v
    return remapped 



def load_txt(filename : str = 'filename'):
    import ast

    with open(filename, 'r') as f:
        lines = f.readlines()

    Headers = "Control points", "Weights", "Patch vertices", "Patch connectivity", \
              "Number of elements per boundary", "Knot values per boundary", \
              "Knot multiplicity per boundary", "Boundary names" # Ordering is of importance! Should be same as in the save_txt() function
    
    # Strip empty spaces and remove '\n'
    lines = [idat.strip("\n") for idat in lines if len(idat.strip("\n")) != 0]   
    catch_line = "View"
    idx = []   
    for i, line in enumerate(lines):
        if line in Headers:
            idx += [i]
    idx += [None]
    loadedData = []
    for k, (i,j) in enumerate(zip(idx[:-1],idx[1:])):
        if k < 3: # We encounter np.arrays
            loadedData += [np.array([ast.literal_eval(iline) for iline in lines[(i+1):j]])]  
        elif k == 3: # We have the weights list (special case)
            loadedData += [np.array([ast.literal_eval(', '.join(lines[(i+1):j]))])]  
        else: # We encounter dicts
            d = {}
            for b in lines[(i+1):j]:
                i = b.split(':')
                if k != len(Headers)-1:
                    d[ast.literal_eval(i[0])] = ast.literal_eval(i[1])
                else: # Else we have the boundaries dict
                    d[i[0]] = i[1]      
            loadedData += [d]

    return loadedData
    


def save_txt(cps : np.ndarray, w : np.ndarray, patches : np.ndarray, patchverts : np.ndarray, 
              knotval : dict, knotmult: dict, nelems: dict, boundaries: dict, 
              filename : str = "filename", headerName : str = "Ventricle"):
    
    if filename.split(".")[-1] != "txt":
        filename += ".txt"

    Title   = f"{headerName} multipatch geometry data"
    Data    = (cps, w, patchverts, patches, nelems, knotval, knotmult, boundaries)
    Headers = "Control points", "Weights", "Patch vertices", "Patch connectivity", \
              "Number of elements per boundary", "Knot values per boundary", \
              "Knot multiplicity per boundary", "Boundary names"

    with open(filename+".txt", 'w') as f:
       f.write(Title+"\n\n")
       for head, data in zip(Headers,Data):
           f.write(head+"\n")
           if type(data) == np.ndarray:
               lines = "\n".join( [ str(row.tolist()) for row in data ] )
           f.write(lines) 
 
           if type(data) == dict:
              for key, value in data.items(): 
                  f.write('%s:%s\n' % (key, value))
           f.write("\n\n") 
    return




if __name__ == "__main__":
   import os
   geomVar = True
   files  = "LV_GEOMETRY_DATA"
   names  = "Left ventricle" 
   folder_read = "examples/output/txt"
   file = "LV_GEOMETRY_DATA.txt"
   data = load_txt(os.path.join(folder_read,file)) 
