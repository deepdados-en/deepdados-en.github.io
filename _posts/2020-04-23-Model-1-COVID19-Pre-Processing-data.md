---
layout: post
title: Model 1 - Tutorial 1 - Automatic detection of COVID-19 cases from chest X-ray images
subtitle: Data pre-processing - Model 1
tags: [COVID]
---

**Being translated!** 23/04/2020

**Main objective of the project:** Automate the process of detecting COVID-19 cases from chest radiograph images, using convolutional neural networks (CNN) through deep learning techniques. The complete project can be accessed [here](https://)

**Steps to reach the goal:**<br />
1- Data pre-processing<br />
[2- Model training and exposure of results](https://deepdados-en.github.io/2020-04-24-Model-1-COVID19-Training-and-Results/)


**Step 1 - Data pre-processing**

*Databases used:*<br />
- X-ray images and chest CT scans of individuals infected with COVID-19 (COHE; MORRISON; DAO, 2020): [link](https://github.com/ieee8023/covid-chestxray-dataset)<br />
- Images of lungs of individuals without any infection (KERMANY; ZHANG; GOLDBAUM, 2018): [link](https://data.mendeley.com/datasets/rscbjbr9sj/2)<br />

*Packages used:*<br />
- Pandas<br />
- Os<br />
- PIL<br />
- Numpy<br />
- CV2<br />

*Code used in the project:*<br />
The notebook with all the codes used in this step is available [here](https://github.com/deepdados-en/ProjetoCOVID/blob/master/preProcessamento_COVID_modelo1_en.ipynb)<br />
**Note:** the numbering and title of each step described in this tutorial correspond with the numbering and title contained in the notebook.

*Steps to be followed:*<br />
**1º Step** – [Import the libraries to be used](#import-the-libraries-to-be-used)<br />
**2º Step** – [Load the dataframe for lung images of individuals with COVID-19](#load-the-dataframe-for-lung-images-of-individuals-with-covid-19)<br />
**3º Step** – [Analysis of the "df" dataframe](#analysis-of-the-df-dataframe)<br />
**4º Step** – [Select the cases related to COVID-19 on the “df” dataframe](#select-the-cases-related-to-covid-19-on-the-df-dataframe)<br />
**5º Step** – [Analysis of the “df_covid” dataframe](#analysis-of-the-df_covid-dataframe)<br />
**6º Step** – [Create a list to add the values of the variable/column “filename”](#create-a-list-to-add-the-values-of-the-variablecolumn-filename)<br />
**7º Step** – [Create a list with only the image formats that exist in the image folder](#create-a-list-with-only-the-image-formats-that-exist-in-the-image-folder)<br />
**8º Step** – [Create a function to open the images, check their dimensions and, later, save this data in a dataframe](#create-a-function-to-open-the-images-check-their-dimensions-and-later-save-this-data-in-a-dataframe)<br />
**9º Step** – [Create a variable that contains as value the address of the folder where the images are saved](#create-a-variable-that-contains-as-value-the-address-of-the-folder-where-the-images-are-saved)<br />
**10º Step** – [Use the function created to check the size of the images](#use-the-function-created-to-check-the-size-of-the-images)<br />
**11º Step** – [Convert all images to 237 x 237px .png](#convert-all-images-to-237-x-237px-png)<br />
**12º Step** – [Create a list of the images that will be deleted from the folder](#create-a-list-of-the-images-that-will-be-deleted-from-the-folder)<br />
**13º Step** – [Open the lung images of individuals without infection and create a list with the name of the images that exist in the image folder](#open-the-lung-images-of-individuals-without-infection-and-create-a-list-with-the-name-of-the-images-that-exist-in-the-image-folder)<br />
**14º Step** – [Convert all images of lungs from uninfected individuals to 237 x 237px .png](#convert-all-images-of-lungs-from-uninfected-individuals-to-237-x-237px-png)<br />
**15º Step** – [Open the images of the lungs of individuals infected with COVID-19 in a list and transform them into an array (matrix of pixel values that represent the image)](#open-the-images-of-the-lungs-of-individuals-infected-with-covid-19-in-a-list-and-transform-them-into-an-array-matrix-of-pixel-values-that-represent-the-image)<br />
**16º Step** – [Open the images of the lungs of individuals without infections in a list and transform them into an array (Matrix of values of the pixels that represent the image)](#open-the-images-of-the-lungs-of-individuals-without-infections-in-a-list-and-transform-them-into-an-array-matrix-of-values-of-the-pixels-that-represent-the-image)<br />
**17º Step** – [Group arrays into a single array containing information about COVID-19 and normal images](#group-arrays-into-a-single-array-containing-information-about-covid-19-and-normal-images)<br />
**18º Step** – [Indicate which cases are COVID-19 and which are normal and create an array](#indicate-which-cases-are-covid-19-and-which-are-normal-and-create-an-array)<br />
**19º Step** – [Save arrays to .npy](#save-arrays-to-npy)<br />

**Tutorial 1:**

**1º Step** 
#### Import the libraries to be used

We imported the Pandas, Os, PIL, Numpy, and CV2 libraries as we will rely on them to pre-process the model data for COVID-19.

``` python
import pandas as pd
import os
from PIL import Image
import numpy as np
import cv2
```

**Note:** the “Pandas” library was imported as “pd”, in order to speed up the writing of the code. That is, instead of typing "pandas" when using it, I will just type "pd". The same was done with the “numpy” library. In addition, the "PIL" library has not been imported completely, as we will not use all the functions contained therein. In this way, it facilitates the use of the library and the processing of codes/data.

**2º Step**
#### Load the dataframe for lung images of individuals with COVID-19

We load the file in .csv, called “metadata”, which accompanies the image bank provided by the researchers (COHE; MORRISON; DAO, 2020).<br />
<br />
The command below names this dataframe “df” when loading it. In parentheses, you must enter the address of this file.

``` python
df = pd.read_csv("/Users/Neto/Desktop/Aprendizados/2020/Kaggle/corona_deep_learning/covid-chestxray-dataset-master/metadata.csv")
```

**3º Step**
#### Analysis of the "df" dataframe

We generated some descriptive data in order to find out how many images of COVID-19 are available on the dataframe (df). For that, we ask for a count of values from the variable/column “finding”. This variable contains the diagnosis related to each lung image.

``` python
df.finding.value_counts()

COVID-19          188
Streptococcus      17
Pneumocystis       15
SARS               11
E.Coli              4
ARDS                4
COVID-19, ARDS      2
Chlamydophila       2
No Finding          2
Legionella          2
Klebsiella          1
Name: finding, dtype: int64
```
It is possible to notice from the data that 188 images refer to COVID-19.<br />

**4º Step**
#### Select the cases related to COVID-19 on the “df” dataframe

We separated only the cases of the variable/column "finding" in the dataframe "df" that were COVID-19 since we will only use these cases in the model. We saved this selection in a new dataframe named “df_covid”.

``` python
df_covid = df[df["finding"] == "COVID-19"]
```

**5º Step**
#### Analysis of the “df_covid” dataframe

We asked to observe the “df_covid” dataframe, in order to analyze whether the selection of COVID-19 cases was carried out correctly. For this, we ask to see the end of this dataframe. In addition, we request that only the variables/columns “finding” and “filename” be shown. The "finding" refers to the selected COVID-19 cases and the "filename" indicates the name of the COVID-19 radiography images made available by the authors of the bank in question (COHE; MORRISON; DAO, 2020). This last information was requested, as it will be used in the next step.

``` python
df_covid[["finding","filename"]].tail()
	finding	filename
307	COVID-19	covid-19-pneumonia-58-day-9.jpg
308	COVID-19	covid-19-pneumonia-58-day-10.jpg
309	COVID-19	covid-19-pneumonia-mild.JPG
310	COVID-19	covid-19-pneumonia-67.jpeg
311	COVID-19	covid-19-pneumonia-bilateral.jpg
```

**6º Step**
#### Create a list to add the values of the variable/column “filename”

We created a list from the variable/column “filename” located in the dataframe “df_covid”. This was called “imagesCOVID”. This list only shows the names of the images with the lungs of individuals infected with the COVID-19 virus. This list was created to facilitate the selection of the images that we will use to train the model.

``` python
imagensCOVID = df_covid["filename"].tolist()
```

**7º Step**
#### Create a list with only the image formats that exist in the image folder

When manually checking the folder where the images are located, only the formats ".jpg" and ".png" were noticed. However, the variable/column "filename" has among its values, images with extension ".gz". Thus, we created a list (“imagensCovid”) with only the name of the images in the formats existing in the folder (“.jpg” and “.png”).

``` python
imagensCovid = []
for imagem in imagensCOVID:
    if imagem.endswith(".gz"):
        pass
    else:
        imagensCovid.append(imagem)
        
print(len(imagensCovid))
```

**8º Step**
#### Create a function to open the images, check their dimensions and, later, save this data in a dataframe

Knowing that this action will be used frequently in the data pre-processing steps of the models that will be trained, we created a function to facilitate this process. Thus, the function below (“df_dimensao”) defines the creation of a dataframe with the dimensions of the images located in a given folder.

``` python
def df_dimensao(folder_das_imagens, lista_nome_imagens):
    """Function to create a dataframe with the original dimensions of the images in a given folder.
    
    Parameters:
    
    folder_das_imagens(str): colocar a pasta onde as imagens estão salvas
    lista_nome_imagens(list): colocar a lista com o nome das imagens
    
    return
    
    df_dims(pd.DataFrame)
    
    """
    
    dic = {}
    dimensaoImagensLargura = []
    dimensaoImagensAltura = []
    nome = []
    
    if ".DS_Store" in lista_nome_imagens:
        lista_nome_imagens.remove(".DS_Store")
    for imagem in lista_nome_imagens:
        
        enderecoDaImagem = folder_das_imagens + "/" + imagem
        abrirImagem = Image.open(enderecoDaImagem)
        nome.append(imagem)
        dimensaoImagensLargura.append(abrirImagem.size[0])
        dimensaoImagensAltura.append(abrirImagem.size[1])

    dic["nome"] = nome
    dic["largura"] = dimensaoImagensLargura
    dic["altura"] = dimensaoImagensAltura
    df_dims = pd.DataFrame(dic)
    
    return df_dims
```

**9º Step**
#### Create a variable that contains as value the address of the folder where the images are saved

In order to use the function created in Step 8, specifically the parameter "folder_das_imagens (str)", we must have a string variable that indicates the address of the images on the computer. For this, the code below creates a variable (“rootFolder”) indicating this location.

``` python
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images"
```

**Note:** in relation to the other function attribute called “lista_nome_imagens”, we will use the list created in Step 7 (“imagensCovid”).<br />

**10º Step**
#### Use the function created to check the size of the images

From the created function, we save the values in the “dimensao” variable. Below you can see the names of each figure and its dimension (width x height) in pixels.

``` python
dimensao = df_dimensao(rootFolder, imagensCovid)
print(dimensao)
                                                  nome  largura  altura
0    auntminnie-a-2020_01_28_23_51_6665_2020_01_28_...      882     888
1    auntminnie-b-2020_01_28_23_51_6665_2020_01_28_...      880     891
2    auntminnie-c-2020_01_28_23_51_6665_2020_01_28_...      882     876
3    auntminnie-d-2020_01_28_23_51_6665_2020_01_28_...      880     874
4                                nejmc2001573_f1a.jpeg     1645    1272
..                                                 ...      ...     ...
211                    covid-19-pneumonia-58-day-9.jpg     2267    1974
212                   covid-19-pneumonia-58-day-10.jpg     2373    2336
213                        covid-19-pneumonia-mild.JPG      867     772
214                         covid-19-pneumonia-67.jpeg      492     390
215                   covid-19-pneumonia-bilateral.jpg     2680    2276

[216 rows x 3 columns]
```

**Note:** this step is important, because to execute the model, all images must have the same dimension.<br />

**11º Step**
#### Convert all images to 237 x 237px .png

Since to run the model we need to have all images with the same dimension, we chose to reduce them all to the dimension of the smallest figure available in the image bank. In addition, to maintain a standard, we changed the format to ".png" for all figures, since some were ".jpg".

The code below resizes the images to 237 x 237px, saves them in another folder and executes the function that we built in Step 8 to see if all dimensions have been changed. 

``` python
for imagem in imagensCovid:
    enderecoDaImagem = rootFolder + "/" + imagem
    abrirImagem = Image.open(enderecoDaImagem)
    image_resize = abrirImagem.resize((237,237))
    os.chdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
    image_resize.save(f'{imagem}_resize_237_237.png')
    
    
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize"
imagensDaPastaResize = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
df_redimensao = df_dimensao(rootFolder, imagensDaPastaResize)
print(df_redimensao)
                                                  nome  largura  altura
0    01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg_resi...      237     237
1    39EE8E69-5801-48DE-B6E3-BE7D1BCF3092.jpeg_resi...      237     237
2                 lancet-case2b.jpg_resize_237_237.png      237     237
3             nejmoa2001191_f4.jpeg_resize_237_237.png      237     237
4    7C69C012-7479-493F-8722-ABC29C60A2DD.jpeg_resi...      237     237
..                                                 ...      ...     ...
211  23E99E2E-447C-46E5-8EB2-D35D12473C39.png_resiz...      237     237
212  covid-19-pneumonia-43-day2.jpeg_resize_237_237...      237     237
213    radiol.2020201160.fig6b.jpeg_resize_237_237.png      237     237
214  8FDE8DBA-CFBD-4B4C-B1A4-6F36A93B7E87.jpeg_resi...      237     237
215      covid-19-pneumonia-7-L.jpg_resize_237_237.png      237     237

[216 rows x 3 columns]
```

**Note:** as you can see, all figures have the same dimension (width x height).<br />

**12º Step**
#### Create a list of the images that will be deleted from the folder

We created a list with the name of the images that were deleted from the folder. The authors of this model decided not to include the lateral and computed tomography images existing in the original image bank. Thus, the variable “listaImagemDeletar” presents a list with the name of these images as a value.

``` python
listaImagemDeletar = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/deletadas")
listaImagemDeletar = ['covid-19-pneumonia-30-L.jpg_resize_237_237.png',
 '396A81A5-982C-44E9-A57E-9B1DC34E2C08.jpeg_resize_237_237.png',
 'covid-19-infection-exclusive-gastrointestinal-symptoms-l.png_resize_237_237.png',
 'nejmoa2001191_f3-L.jpeg_resize_237_237.png',
 '3ED3C0E1-4FE0-4238-8112-DDFF9E20B471.jpeg_resize_237_237.png',
 'covid-19-pneumonia-38-l.jpg_resize_237_237.png',
 'a1a7d22e66f6570df523e0077c6a5a_jumbo.jpeg_resize_237_237.png',
 '254B82FC-817D-4E2F-AB6E-1351341F0E38.jpeg_resize_237_237.png',
 'covid-19-pneumonia-15-L.jpg_resize_237_237.png',
 'kjr-21-e24-g002-l-b.jpg_resize_237_237.png',
 'D5ACAA93-C779-4E22-ADFA-6A220489F840.jpeg_resize_237_237.png',
 'kjr-21-e24-g002-l-c.jpg_resize_237_237.png',
 'covid-19-pneumonia-14-L.png_resize_237_237.png',
 'kjr-21-e24-g004-l-a.jpg_resize_237_237.png',
 'nejmoa2001191_f1-L.jpeg_resize_237_237.png',
 'kjr-21-e24-g003-l-b.jpg_resize_237_237.png',
 'kjr-21-e24-g004-l-b.jpg_resize_237_237.png',
 'DE488FE1-0C44-428B-B67A-09741C1214C0.jpeg_resize_237_237.png',
 '191F3B3A-2879-4EF3-BE56-EE0D2B5AAEE3.jpeg_resize_237_237.png',
 '35AF5C3B-D04D-4B4B-92B7-CB1F67D83085.jpeg_resize_237_237.png',
 '6A7D4110-2BFC-4D9A-A2D6-E9226D91D25A.jpeg_resize_237_237.png',
 '4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg_resize_237_237.png',
 'covid-19-caso-70-1-L.jpg_resize_237_237.png',
 '44C8E3D6-20DA-42E9-B33B-96FA6D6DE12F.jpeg_resize_237_237.png',
 'kjr-21-e24-g001-l-b.jpg_resize_237_237.png',
 'FC230FE2-1DDF-40EB-AA0D-21F950933289.jpeg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-a.jpg_resize_237_237.png',
 '925446AE-B3C7-4C93-941B-AC4D2FE1F455.jpeg_resize_237_237.png',
 'jkms-35-e79-g001-l-e.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-b.jpg_resize_237_237.png',
 '21DDEBFD-7F16-4E3E-8F90-CB1B8EE82828.jpeg_resize_237_237.png',
 'covid-19-pneumonia-evolution-over-a-week-1-day0-L.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-d.jpg_resize_237_237.png',
 '1-s2.0-S0929664620300449-gr3_lrg-c.jpg_resize_237_237.png',
 'nejmoa2001191_f5-L.jpeg_resize_237_237.png',
 'jkms-35-e79-g001-l-d.jpg_resize_237_237.png',
 'covid-19-pneumonia-22-day1-l.png_resize_237_237.png',
 'kjr-21-e24-g001-l-c.jpg_resize_237_237.png',
 '66298CBF-6F10-42D5-A688-741F6AC84A76.jpeg_resize_237_237.png',
 'covid-19-pneumonia-20-l-on-admission.jpg_resize_237_237.png',
 'covid-19-pneumonia-7-L.jpg_resize_237_237.png']
```

**13º Step**
#### Open the lung images of individuals without infection and create a list with the name of the images that exist in the image folder

After creating a variable called “pastaTreinoNormal” with the address of the folder with the images of the lungs of individuals without infection, we created a list (“listaImagensTreino”) with only the name and format of these images.

``` python
pastaTreinoNormal = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL"

listaImagensTreino = os.listdir(pastaTreinoNormal)
```

**14º Step**
#### Convert all images of lungs from uninfected individuals to 237 x 237px .png

The images of normal lungs were resized to the same dimension as the lung images with COVID-19: namely, 237 x 237px. To maintain the same pattern, we changed the format to “.png” for all figures. It is important to note that we selected only the first 100 images in the folder using the code below. This was done to maintain the training with a similar amount of image of individuals without any infection and with COVID-19.

In addition, we run the function we built in Step 8 to see if all dimensions have changed.

``` python
listaCemImagens = listaImagensTreino[0:100]
for imagem in listaCemImagens:
    enderecoDaImagem = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL"+ "/" + imagem
    abrirImagem = Image.open(enderecoDaImagem)
    image_resize = abrirImagem.resize((237,237))
    os.chdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")
    image_resize.save(f'{imagem}_resize_237_237.png')
  
rootFolder = "/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal"
imagensDaPastaResize = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")
df_redimensao = df_dimensao(rootFolder, imagensDaPastaResize)
print(df_redimensao)
                                            nome  largura  altura
0   NORMAL2-IM-1196-0001.jpeg_resize_237_237.png      237     237
1   NORMAL2-IM-0645-0001.jpeg_resize_237_237.png      237     237
2           IM-0269-0001.jpeg_resize_237_237.png      237     237
3   NORMAL2-IM-1131-0001.jpeg_resize_237_237.png      237     237
4      IM-0545-0001-0002.jpeg_resize_237_237.png      237     237
..                                           ...      ...     ...
95  NORMAL2-IM-0592-0001.jpeg_resize_237_237.png      237     237
96  NORMAL2-IM-1167-0001.jpeg_resize_237_237.png      237     237
97  NORMAL2-IM-0741-0001.jpeg_resize_237_237.png      237     237
98  NORMAL2-IM-0535-0001.jpeg_resize_237_237.png      237     237
99          IM-0119-0001.jpeg_resize_237_237.png      237     237

[100 rows x 3 columns]
```

**Note:** as you can see, all figures have the same dimension (width x height).<br />

**15º Step**
#### Open the images of the lungs of individuals infected with COVID-19 in a list and transform them into an array (matrix of pixel values that represent the image)

First, from the resized lung images of individuals with COVID-19 obtained in Step 11, we created a variable (“imagensCovid”) with the list of names of these images. Then, using the list of images that were not used in the model (lateral and computed tomography), referring to Step 12, these were deleted from the variable values (“imagensCovid”).

Subsequently, we created a list with the arrays called “XTrainCovid” from the resized images, that is, a list with the values referring to the pixels that represent the lungs of individuals infected by COVID-19.

Finally, we saved the “XTrainCovid” list in an array called “xArrayCOVID”.

``` python
imagensCovid = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize")
imagensCovid = [x for x in imagensCovid if x not in listaImagemDeletar]

if ".DS_Store" in imagensCovid:
    imagensCovid.remove(".DS_Store")

xTrainCovid = []

for image in imagensCovid:
    x = cv2.imread("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/covid-chestxray-dataset-master/images/images_resize/" + image)
    x = np.array(x)
    xTrainCovid.append(x)

xArrayCOVID = np.array(xTrainCovid)
print(xArrayCOVID.shape)

(175, 237, 237, 3)
```

**Note:** as you can see, the built array (“xArrayCOVID”) has four dimensions. The first (“175”) refers to the number of cases, that is, images of individuals with COVID-19; the second (“237”) refers to the width of the image; the third (“237”) refers to the height of the image and; the fourth (“3”), the number of color channels in the images.<br />

**16º Step**
#### Open the images of the lungs of individuals without infections in a list and transform them into an array (Matrix of values of the pixels that represent the image)

First, from the resized images of the lungs of individuals without infections obtained in Step 13, we created a variable (“imagensNormal”) with the list of names of these images.

In a second step, we created a list with the arrays called “XTrainNormal” from the resized images, that is, a list with the values referring to the pixels that represent the lungs of individuals without infections.

Finally, we save the “XTrainNormal” list in an array called “xArrayNormal”.

``` python
imagensNormal = os.listdir("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal")

if ".DS_Store" in imagensNormal:
    imagensNormal.remove(".DS_Store")

xTrainNormal = []

for image in imagensNormal:
    x = cv2.imread("/Users/cesarsoares/Documents/Python/COVID/Banco_de_Dados/chest_xray/train/NORMAL/images_resize_normal/" + image)
    x = np.array(x)
    xTrainNormal.append(x)

xArrayNormal = np.array(xTrainNormal)
print(xArrayNormal.shape)

(100, 237, 237, 3)
```

**Note:** as you can see, the built array (“xArrayNormal”) has four dimensions. The first (“100”) refers to the number of cases, that is, images of individuals without infections; the second (“237”) refers to the width of the image; the third (“237”) refers to the height of the image and; the fourth (“3”), the number of color channels in the images.<br />

**17º Step**
#### Group arrays into a single array containing information about COVID-19 and normal images

We group the array of images of individuals with COVID-19 (“xArrayCOVID”) created in Step 14 with the array of images of individuals without infections (“xArrayNormal”) created in Step 15. This array was saved in the variable “X_train”.

``` python
X_train = np.vstack((xArrayCOVID,xArrayNormal))
```

**18º Step**
#### Indicate which cases are COVID-19 and which are normal and create an array

The “dfCOVID” variable created added the value “1” in the 175 lines indicating the presence of COVID-19. And the variable "dfNormal", added the value "0" in the 100 lines pointing the images of lungs of individuals without infections.

Finally, we group the array of images of individuals with COVID-19 (“dfCOVID”) with the array of images of individuals without infections (“dfNormal”). This array was saved in the “Y_train” variable.

``` python
dfCOVID = np.ones((xArrayCOVID.shape[0],1))
dfNormal = np.zeros((xArrayNormal.shape[0],1))

Y_train = np.vstack((dfCOVID,dfNormal))
```

**19º Step**
#### Save arrays to .npy

To use the arrays in training the model, these were saved in “X_Train.npy” and “Y_Train.npy”.

``` python
np.save("/Users/cesarsoares/Documents/Python/COVID/X_Train.npy",X_train)
np.save("/Users/cesarsoares/Documents/Python/COVID/Y_Train.npy", Y_train)
```

**Note:** X_Train will be the input of the trained model and Y_Train will be the target, that is, the expected result of the model.<br />

**Bibliography** <br />
COHEN, Joseph; MORRISON, Paul; DAO, Lan. COVID-19 Image Data Collection. arXiv:2003.11597, 2020.<br />
<br />
KERMANY, Daniel; ZHANG, Kang; GOLDBAUM, Michael. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v.2, 2018. Disponível em: http://dx.doi.org/10.17632/rscbjbr9sj.2
