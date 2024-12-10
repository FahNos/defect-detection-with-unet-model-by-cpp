import sys
from tensorflow import keras
import tensorflow as tf
import gguf

def convert(model_name):
    model = keras.models.load_model(model_name, compile=False)
    gguf_model_name = model_name + ".gguf"
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "Unet")
    for layer in model.layers:      
        # export layers with weights
        if layer.weights:
            print(f"   Layer has {len(layer.weights)} weights")
            for weight in layer.weights:
                print(f"  [{weight.name}] {weight.shape} {weight.dtype}")
         
                weight_data = weight.numpy()
               
                if(len(weight.shape) == 1):
                    weight_data = weight_data.reshape(1, 1, -1, 1)
                    print(f"  after transpose: [{weight.name}] {weight_data.shape} {weight.dtype}")
                gguf_writer.add_tensor(weight.name, weight_data.T)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Model converted and saved to '{}'".format(gguf_model_name))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        model_file = "modelunet.h5"

    convert(model_file)
