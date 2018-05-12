def convert_to_pb(weight_file,json_file,input_fld='',output_fld=''):

    import os
    import os.path as osp
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from keras.models import model_from_json
    from keras import backend as K
    import tensorflow as tf

    # weight_file is a .hdf5 keras model file
    output_node_names_of_input_network = ["pred0"] 
    output_node_names_of_final_network = 'output_node'

    # change filename to a .pb tensorflow file
    output_graph_name = json_file[:-4]+'pb'
    
    weight_file_path = osp.join(input_fld, weight_file)
    json_file_path=osp.join(input_fld,json_file)
    js_file = open(json_file_path, 'r')
    model_js= js_file.read()
    js_file.close()
    net_model = model_from_json(model_js)
    # load weights into new model
    net_model.load_weights(weight_file_path)
    

    num_output = len(output_node_names_of_input_network)
    pred = [None]*num_output
    pred_node_names = [None]*num_output

    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

    return output_fld+output_graph_name


#tf_model_path = convert_to_pb('generator_weights.hdf5','generator.json','saved_model/','tf_model/')