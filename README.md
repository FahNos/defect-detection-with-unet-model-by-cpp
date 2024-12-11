# Defect detection with unet model

Training Unet model base on resnet50 and apply cpp for deployment base on ggml repository
![result](https://github.com/user-attachments/assets/6bec9aa7-d457-45b6-b0cc-212a1fd40f23)

## Quick start

Download files: [modelunet.gguf](https://huggingface.co/FahNos/defec_detection_model_unet/resolve/main/modelunet.gguf?download=true), [ggml.dll](https://huggingface.co/FahNos/defec_detection_model_unet/resolve/main/ggml.dll?download=true) and [unet.exe](https://huggingface.co/FahNos/defec_detection_model_unet/resolve/main/unet.exe?download=true)

```bash
cd unet folder
unet -i image.jpg
```
Running with CUDA on window os
```bash
cd ggml
cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe
msbuild ALL_BUILD.vcxproj /p:Configuration=Release 
```

## Convert file h5 model to gguf
Request tensorflow 2.15, download file [h5 model](https://huggingface.co/FahNos/defec_detection_model_unet/resolve/main/modelunet.h5?download=true)

```bash
python convert.py modelunet.h5
```
## Training model
Training file Unet_detection.ipynb, download data set [here](https://www.mediafire.com/file/o9u2x1v1n0ffmp5/NV_public_defects.zip/file)
## Run speed
![](https://github.com/user-attachments/assets/f991dc65-8312-4920-b5c4-4da9d22bd6ad)

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## References

- [ggml](https://github.com/ggerganov/ggml)
- [MiAI_Defect_Detection](https://github.com/thangnch/MiAI_Defect_Detection)