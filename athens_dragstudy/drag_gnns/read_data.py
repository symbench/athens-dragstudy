import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import copy
import json
from pandas.io.json import json_normalize
import os, glob
from zipfile import ZipFile
import numpy as np
#from IPython.display import display

IN_CONNECTOR = 1.0
OUT_CONNECTOR = 2.0
df_schema = pd.read_json('data/corpus_schema.json')
df_corpus_data = pd.read_json('data/corpus_data.json')

ZIP_FILES_PATH = "data/random_designs/"
DESIGN_JSON_PATH = "data/drag_study_data/designs/"
###############################################################
############# Get values from text based file in dict ##################
def extract_values(filename, parameters):
    val_dict = {}
        
    # Using readline()
    #file1 = open(filename, 'r')
    file1 = filename
    count = 0
    
    while True:

        # Get next line from file
        line = file1.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break

        line = line.decode('ascii')
        for param in parameters:
            if line.find(param) != -1:
                val_dict[param] = float( line[ (line.find("=")+1): ] )
    
    file1.close()
    return val_dict
###############################################################
###############  Move design files and label data to design ###

# importing required modules
  
def get_design_files_from_zip(path = ZIP_FILES_PATH,
    out_path = DESIGN_JSON_PATH):
    # get all files in the design folder
    # specifying the zip file name
    
    count = 0
    column_names = ["design_name", "Interferences", "MassEstimate", "AnalysisError", "TotalPathScore"]
    column_names = ["x_fuse","y_fuse","z_fuse","X_fuseuu","Y_fusevv","Z_fuseww"]
   

    labels_df = pd.DataFrame(columns = column_names)

    for filename in glob.glob(os.path.join(path, '*.zip')):      
        # opening the zip file in READ mode
        with ZipFile(filename, 'r') as zip:
            # Get list of files names in zip
            listOfiles = zip.namelist()
            design_flag = False
            design_file = None
            output_flag = False
            output_file = None
            # Iterate over the list of file names in given list & print them
            for elem in listOfiles:
                if elem.find('design_data.json') != -1:
                    design_file = elem
                    design_flag = True
# mass properties are in output.csv but drag properties are in input file in archive directory
#                if elem.find('output.csv') != -1:
                if elem.find('archive/result_1/flightDynFast.inp') != -1:
                    output_file = elem
                    output_flag = True
            if design_flag and output_flag:
                
                count += 1
                # read the output to labels dataframe
                drag_dict = extract_values(zip.open(output_file),column_names)

                #df = pd.read_csv(zip.open(output_file))
                
                index = len(labels_df.index)
                labels_df.loc[index, 'design_name'] = design_file#.split('_design_data.json')[0]
                for col in column_names:
                    labels_df.loc[index, col] = drag_dict[col]
                # move design file to a new folder
                zip.extract(design_file, out_path)
                #break
    #print(count)
    labels_df.to_csv(out_path+'labels.csv', index=False)
                    
            # read json design file
            #data = zip.read('design.json')
###############################################################
###############  Create the node attributes list from corpus ##
def get_attributes_list():
    ## Set of all properties and parameters
    # access dataframe properties of columns one by one
    properties = []
    for i in range(len(df_schema.columns)):
        for key, value in df_schema.iloc[:, i].loc['properties'].items():
            if value == 'float' or value == 'int':
                properties.append( key ) 
    #print( len(properties) )
    # access dataframe parameters of columns one by one
    parameters = []
    for i in range(len(df_schema.columns)):
        for key, value in df_schema.iloc[:, i].loc['parameters'].items():
            if value == 'float' or value == 'int':
                parameters.append( key )
    #print( len(parameters) )
    node_attributes = list(set(properties + parameters))
    print( len(node_attributes) )
    # access dataframe connectors of columns one by one
    in_connectors = []
    out_connectors = []
    for i in range(len(df_schema.columns)):
        for key, value in df_schema.iloc[:, i].loc['connectors'].items():
            if value == 'in' and key not in in_connectors:
                in_connectors.append( key )
            elif value == 'out' and key not in out_connectors:
                out_connectors.append( key )

    #print( len(in_connectors) )
    #print( len(out_connectors) )
    edge_attributes = in_connectors + out_connectors
    print( len(edge_attributes) )

    #print( in_connectors )
    #print( out_connectors )

    #TODO: We are adding edge attributes to node attributes. This may need to be changed
    node_attributes = node_attributes + edge_attributes
    node_attributes.append( 'is_connector' )
    node_attributes.append( 'connector_type' )
    
    return node_attributes

# Search in the instances list for the model name
def get_instance_by_name(node_instances, name):
    for node_item in node_instances:
        if node_item['name'] == name:
            return node_item
    return None
def get_connector_type(v_parts, node_instances, df_corpus_data,df_schema):
    # check in schema if v is an in connector
    v_instance = v_parts[0]
    v_connector = v_parts[1]
    # get model name from instances then search for class name in corpus
    v_model = get_instance_by_name(node_instances, v_instance)['model']
    v_class =  df_corpus_data.loc[(df_corpus_data['model'] == v_model)]['class'].tolist()[0]
    return IN_CONNECTOR if df_schema[v_class]['connectors'][v_connector]=='in' else OUT_CONNECTOR

    # 0 is used when it is not a connector

################################################################
###############  Read the design file and create graph##########
def read_graph_from_design(design_file = 'designs/TailSitter3.json'):
    with open(design_file) as json_data:
        data = json.load(json_data)
    df_design = data['connections']
    #df_design = pd.read_json(design_file)
    edges = []
    for item in df_design:
        # insteading just adding one edge, we add three edges to include connectors:
        #e = (item['instance1'], item['instance2'])
        e = (item['instance1'], item['instance1']+"$"+item['connector1'])    
        edges.append(e)
        e = (item['instance1']+"$"+item['connector1'], item['instance2']+"$"+item['connector2'])
        edges.append(e)
        e = (item['instance2']+"$"+item['connector2'], item['instance2'])
        edges.append(e)
    G = nx.Graph()
    G.add_edges_from(edges)
    #TODO: add attributes to nodes using new format
    
    #nx.draw_networkx(G, with_labels=False)
    ###############################################################
    ###############  Create the node attributes ###################
    node_attributes = get_attributes_list()
    assigned_attributes = data['parameters']
    node_instances = data['instances']
    attributes = {}
    for v in G.nodes():
        attibute_dict = {}
        v_parts = v.split('$')
        # if v is a connector then there are only three meaningful attributes
        # connector type (IN or OUT), is_connector and connector name
        if len(v_parts) == 2:
            for a in node_attributes:
                attibute_dict[a] = 0.0
            
            attibute_dict['is_connector'] = 1.0
            # check in schema if v is an in connector
            attibute_dict['connector_type'] = get_connector_type(v_parts, node_instances, df_corpus_data,df_schema)
            # Set one hot encoding for connector name
            attibute_dict[v_parts[1]] = 1.0
        else:
        
            for a in node_attributes:
                attibute_dict[a] = 0.0
                # search v in df instaces
                node_item = get_instance_by_name(node_instances, v)
                if a in node_item['assignment']:
                    value_str = node_item['assignment'][a]
                    attibute_dict[a] = float( assigned_attributes[value_str] )
                    
                else:
                    model_name = node_item['model']
                    # search model in df corpus
                    
                    df_corpus_data.reset_index(drop=True, inplace=True)
                    model_corpus = df_corpus_data.loc[ df_corpus_data['model'] == model_name ]
                    model_corpus.reset_index(drop=True, inplace=True)
                    #model_corpus = model_corpus[0]

                    if a in model_corpus['parameters'][0].keys():
                        attibute_dict[a] = float( model_corpus['parameters'][0][a]['assigned'] )
                    
                    elif a in model_corpus['properties'][0].keys():
                        attibute_dict[a] = float( model_corpus['properties'][0][a] )
                    
        attributes[v] = copy.deepcopy(attibute_dict)
    nx.set_node_attributes(G, attributes)
    return G, attributes
    


    #filename = "/home/mudassir/work/peter/vudoruns/designs/VariCylLengthA3_design_data.json"
    #G, attributes = read_graph_from_design(filename)
    #print ( G.number_of_nodes() )
import pickle 

#unit_size = 50
#nx.draw_networkx(G, with_labels=False, nodelist=d.keys(), node_size=[v * unit_size for v in d.values()])
#print(nx.info(G))
def write_data_to_pkl():

    get_design_files_from_zip()

    labels_df = pd.read_csv( DESIGN_JSON_PATH + 'labels.csv')
    print("labels: ",labels_df.shape)
    graphs = []
    attributes = []
    labels = []

    NUMBER_OF_ATTRIBUTES = len(get_attributes_list())
    LABEL_NAME = 'MassEstimate'
    LABEL_NAME = 'x_fuse'
    
    count = 0
    for elem in labels_df.iterrows():
        graph, attribute = read_graph_from_design( DESIGN_JSON_PATH + elem[1]['design_name'])
        
        #True number of atrributes is usually different
        NUMBER_OF_ATTRIBUTES = len(list(attribute.values())[0].values())
        
        graphs.append(graph)
        attr_matrix = np.zeros( (len(graph.nodes()), NUMBER_OF_ATTRIBUTES) )
        for i, node in enumerate(graph.nodes()):
            #for j, key in enumerate(attribute.keys()):
            att_dict = attribute[node].values()
            attr_matrix[i] = list(att_dict)
        attributes.append(attr_matrix)
        labels.append( elem[1][LABEL_NAME] )
        print(count)
        count += 1

    data = {'graphs': graphs, 'attributes': attributes, 'labels': labels}
    file_pi = open(DESIGN_JSON_PATH+'data.pkl', 'wb') 
    pickle.dump(data, file_pi)
    file_pi.close()

def read_data_from_pkl():
    file_pi = open(DESIGN_JSON_PATH+'data.pkl', 'rb') 
    data = pickle.load(file_pi)
    return data

write_data_to_pkl()
data = read_data_from_pkl()
labels = data['labels']
#plot(labels)
# plot labels
import matplotlib.pyplot as plt
sorted_labels = sorted(labels)
plt.plot(sorted_labels)
plt.show()
print(sorted_labels)


