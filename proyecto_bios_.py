from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def num_bin_aleatorio():
    binario =[str(x) for x in random.choices([0,1], k=9)]
    binari=''.join(binario)
    return binari

def limpieza_de_vacio(X_train,X_test):
    #busca signos ? y los cambia por np.nan y los que son numeros 
    #en str los vuelve int  para el entrenamiento
    for i in range(len(X_train)):
        for j in range(len(X_train[0])):
            aux=str(type(X_train[i][j]))
            if aux=="<class 'str'>":
                if X_train[i][j]=='?':
                    X_train[i][j]=np.nan
                else:
                    X_train[i][j]=int(X_train[i][j])
    #hace lo mismo para el test
    for i in range(len(X_test)):
        for j in range(len(X_test[0])):
            aux=str(type(X_test[i][j]))
            if aux=="<class 'str'>":
                if X_test[i][j]=='?':
                    X_test[i][j]=np.nan
                else:
                    X_test[i][j]=int(X_test[i][j])
    #cambia los np.nan por una media de lo de mas           
    data_new = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_new.fit(X_train)
    X_train=data_new.transform(X_train)
    #hace lo mismo pero en el test 
    data_new = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_new.fit(X_test)
    X_test=data_new.transform(X_test)
    #retorna los datos ya limpios 
    return X_train,X_test

def Probabilidades(lista):
    listafit=[]
    for i in range(len(lista)):
        listafit.append(lista[i][1])
    total=sum(listafit)
    listaproba=[]
    for i in listafit:
            listaproba.append(i/total)
    aux=0
    listaprobAcum=[]
    for i in listaproba:
        aux+=i
        listaprobAcum.append(aux)
    return listaproba,listaprobAcum

def seleciontwopadres(listaacu):
    
    bandera=0
    valoresaleatorios=[random.random(),random.random()]
    while(bandera==0):
        posicionespadres=[]
        valoresaleatorios.sort()
        for i in valoresaleatorios:
            for j in listaacu:
                if (i<j):
                    posicionespadres.append(listaacu.index(j))
                    break
        if(posicionespadres[0]!=posicionespadres[1]):
            bandera=1
        else:
            valoresaleatorios[1]=random.random()

    return posicionespadres

def seleccion_de_caracteristicas(binario,X):

    lista=[]
    real=['Clump Thickness ','Uniformity of Cell Size', ' Uniformity of Cell Shape ', 'Marginal Adhesion', 'Single Epithelial Cell Size  ', 'Bare Nuclei ','Bland Chromatin','Normal Nucleoli','Mitoses']
    for i in range(len(binario)):
        if binario[i]=='0':
            lista.append(i)
    lista.sort(reverse=True)
    for i in lista:
        name=real[i]
        X=X.drop([name], axis=1)

    return X
            

def cruza(listade2padres):
    hijos=[]
    gen=listade2padres
    y=random.randint(1,8)

    cadena11=gen[0][:y]
    cadena12=gen[0][y:]   
    
    cadena21=gen[1][:y]
    cadena22=gen[1][y:]
    hijos.append(cadena11+cadena22)
    hijos.append(cadena21+cadena12)
    
    return hijos

def mutacion(hijos):
    hijos=hijos
    hijo1=hijos[0]
    hijo2=hijos[1]
    hijos.pop()
    hijos.pop()
    p_m=0.1
    for index, valor in enumerate(hijo1):
        num=random.random()
        if(p_m>num):
            if (valor=='0'):
                valorNew='1'
                hijo1 = hijo1[:index]+valorNew+hijo1[index+1:]
            elif (valor=='1'):
                valorNew1='0'
                hijo1 = hijo1[:index]+valorNew1+hijo1[index+1:]
        
    for index, valor in enumerate(hijo2):
        num=random.random()
        if(p_m>num):
            if(valor=='0'):
                valorNew2='1'
                hijo2 = hijo2[:index]+valorNew2+hijo2[index+1:]
            elif (valor=='1'):
                valorNew3='0'
                hijo2 = hijo2[:index]+valorNew3+hijo2[index+1:]
    hijos.append(hijo1)
    hijos.append(hijo2)
    
    return hijos

def funcionfit(dat,binario):

    data_copy=dat.copy()
    binario=binario
    X=seleccion_de_caracteristicas(binario,dat)
    X=X.values
    Y = data['class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train,X_test=limpieza_de_vacio(X_train,X_test)

    svc_model = SVC(gamma='scale')
    svc_model.fit(X_train, y_train)
    predictions = svc_model.predict(X_test)
    error= accuracy_score(y_test, predictions)
    peso_extra=0
    for i in range(len(binario)):
        if binario[i]=='1':
            peso_extra=peso_extra+1
    fit=round(error,3)
    
    return binario,fit, peso_extra

# leo el data
data =  pd.read_csv('breast-cancer-wisconsin.data', sep=",")
#coloco columnas
data.columns = ['id number','Clump Thickness ','Uniformity of Cell Size', ' Uniformity of Cell Shape ', 'Marginal Adhesion', 'Single Epithelial Cell Size  ', 'Bare Nuclei ','Bland Chromatin','Normal Nucleoli','Mitoses','class']
#quito columna de tag y de id 
dat=data.drop(['id number','class'],axis=1)
#llamo a la funcion fit 
binario,fit,peso=funcionfit(dat,'111111111')
#creo la lista de individuos y coloco el primero
lista_cromosomas=[]
lista_cromosomas.append([])
lista=lista_cromosomas[0]
lista.append(binario)
lista.append(fit)
lista.append(peso)

#creo 10  individuos mas asegurandose que no sean 0 total 
for j in range(10):
    binario=num_bin_aleatorio()
    bandera=1
    while bandera ==1:
        bandera2=1
        for i in range(len(lista_cromosomas)):
            if(binario == lista_cromosomas[i][0] or binario=='000000000' ):
                binario=num_bin_aleatorio()
                bandera2=0
                break
        if bandera2==1:
            bandera=0
    lista_cromosomas.append([])
    lista=lista_cromosomas[i+1]            
    lista.append(binario)     

#evaluo los 10 individuos  y agrego a la lista de individuos
for i in range(10):
    data_copy=dat.copy()
    binario=lista_cromosomas[i+1][0]
    X=seleccion_de_caracteristicas(binario,dat)
    binario,fit,peso=funcionfit(dat,binario)
    lista=lista_cromosomas[i+1]
    lista.append(fit)
    lista.append(peso)

#ordeno la lista de modo que quede el mas debil abajo 
lista_cromosomas=sorted(lista_cromosomas, key=lambda x:x[2])
lista_cromosomas=sorted(lista_cromosomas, key=lambda x:x[1],reverse=True)
#elimino el mas debil 
lista_cromosomas.pop(-1)
#imprimo mi primer lista de individuos
df = pd.DataFrame(lista_cromosomas)
df.columns=['caracteristicas','accuracy','columnas activas']
print(df)

for i in range(100):
    #calculo probabilidades
    listaproba,listaacu=Probabilidades(lista_cromosomas)

    bandera=0 #esta bandera es por si no se cruza se eligen de nuevo otros dos padres
    while(bandera==0):
        #se eligen padres
        listade2padres=seleciontwopadres(listaacu)
        listade2padres=[lista_cromosomas[listade2padres[0]][0],lista_cromosomas[listade2padres[1]][0]]
        probadecruza=0.85
        #se saca un numero aleatorio
        numalprobacruza=random.random()
        #se evalua la cruza
        if (probadecruza>numalprobacruza):
            #se cruza
            listade2hijos=cruza(listade2padres)
            #se muta
            listade2hijos=mutacion(listade2hijos)
            #se juntan y se crea la nueva lista
            nuevalista=listade2padres+listade2hijos

            newlistatratada=[]
            #se evalua  la  nueva lista
            for i in range(len(nuevalista)):
                newlistatratada.append([])
                lista=newlistatratada[i]
                binario,fit,peso=funcionfit(dat,nuevalista[i])
                lista.append(binario)
                lista.append(fit)
                lista.append(peso)
            #se ordena la nueva lista
            newlistatratada=sorted(newlistatratada, key=lambda x:x[2])
            newlistatratada=sorted(newlistatratada, key=lambda x:x[1],reverse=True)
            #se seleciona el primero que es el mejor
            seleccionado=newlistatratada[0]
            #se elimina el ultimo de nuestra lista 
            lista_cromosomas.pop(-1)
            #se agrega el mejor 
            lista_cromosomas.append(seleccionado)
            #se vuelve a ordenar
            lista_cromosomas=sorted(lista_cromosomas, key=lambda x:x[2])
            lista_cromosomas=sorted(lista_cromosomas, key=lambda x:x[1],reverse=True)
            #acaba el while
            bandera=1
            #todo se repite 100 veces

#imprime el resultado 
print('\n\nfinal\n')
df = pd.DataFrame(lista_cromosomas)
df.columns=['caracteristicas','accuracy','columnas activas']
print(df)
 




