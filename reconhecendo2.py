import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import json
import wikipedia
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageTk

model = None
class_indices = None
cap = None
ultima_previsao = None

def buscar_info_wikipedia(animal):
    print(f"Buscando por '{animal}' na Wikipedia...")
    try:
        wikipedia.set_lang("pt")
        
        resumo = wikipedia.summary(animal, sentences=5, auto_suggest=True)
        
        info_api = {
            "destaques": resumo
        }
        return info_api

    except wikipedia.exceptions.PageError:
        print(f"Erro: A página para '{animal}' não foi encontrada na Wikipedia.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Erro: O termo '{animal}' é ambíguo. Opções: {e.options}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao acessar a Wikipedia: {e}")
        return None

def abrir_janela_informacoes(especie):
    
    info = buscar_info_wikipedia(especie)

    if not info:
        messagebox.showinfo("Resultado", f'O animal é: {especie.capitalize()}\n\n(Informações não encontradas na base de dados)')
        return
    
    janela_info = tk.Toplevel()
    janela_info.title(f"Informações sobre: {especie.capitalize()}")
    janela_info.geometry("400x550")
    
    lbl_titulo = tk.Label(janela_info, text=especie.capitalize(), font=("Arial", 16, "bold"))
    lbl_titulo.pack(pady=10)

    frame_stats = tk.Frame(janela_info)
    frame_stats.pack(pady=5, padx=10, fill="x")
    
    def criar_secao(parent, titulo, texto):
        lbl_titulo_secao = tk.Label(parent, text=titulo, font=("Arial", 12, "bold"))
        lbl_titulo_secao.pack(pady=(15, 2), anchor="w", padx=10)
        
        lbl_texto_secao = tk.Label(parent, text=texto, wraplength=380, justify="left", font=("Arial", 10))
        lbl_texto_secao.pack(anchor="w", padx=10)

    criar_secao(janela_info, "Destaques e Curiosidades", info['destaques'])


def verificar_estrutura_pasta(pasta_imagens):
    if not os.path.exists(pasta_imagens): return False
    subpastas = [f.path for f in os.scandir(pasta_imagens) if f.is_dir()]
    if len(subpastas) == 0: return False
    for subpasta in subpastas:
        if len([f for f in os.scandir(subpasta) if f.is_file()]) == 0: return False
    return True

def criar_modelo_com_transfer_learning(numero_de_classes):

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    predictions = Dense(numero_de_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def treinar_modelo(pasta_imagens):
    global model, class_indices
    if not verificar_estrutura_pasta(pasta_imagens):
        messagebox.showerror("Erro de Estrutura", "A pasta selecionada não contém subpastas com imagens. Verifique a estrutura.")
        return
    
    print("Iniciando o treinamento do modelo com Transfer Learning...")
    messagebox.showinfo("Treinamento", "O treinamento foi iniciado. Por favor, aguarde a mensagem de conclusão.")

    datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    train_generator = datagen.flow_from_directory(pasta_imagens, target_size=(224, 224), batch_size=32, class_mode='categorical')

    model = criar_modelo_com_transfer_learning(len(train_generator.class_indices))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=20)
    
    model.save('meu_modelo.h5')
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    acc_final = history.history['accuracy'][-1] * 100
    messagebox.showinfo("Treinamento Concluído", f"Modelo treinado com precisão final de {acc_final:.2f}% e salvo com sucesso!")

def carregar_modelo():
    global model, class_indices
    try:
        model = load_model('meu_modelo.h5')
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        return True
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return False

def reconhecer_objeto(frame, class_indices):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    previsao = model.predict(image)
    classe_index = np.argmax(previsao)
    confianca = np.max(previsao) * 100
    
    indices_para_classes = {v: k for k, v in class_indices.items()}
    nome_classe = indices_para_classes[classe_index]
    
    return nome_classe, confianca

def atualizar_imagem(label_img):
    global cap, ultima_previsao
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if model and class_indices:
                especie, confianca = reconhecer_objeto(frame, class_indices)
                ultima_previsao = especie
                
                texto_previsao = f'{especie.capitalize()} ({confianca:.1f}%)'
                
                cv2.rectangle(frame, (5, 5), (300, 40), (0, 0, 0), -1)

                cv2.putText(frame, texto_previsao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_img.config(image=img_tk)
            label_img.image = img_tk

        label_img.after(10, lambda: atualizar_imagem(label_img))

def ativar_camera(label_img):
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro de Câmera", "Não foi possível acessar a câmera.")
            cap = None
            return
        atualizar_imagem(label_img)

def capturar_e_analisar():
    if ultima_previsao:
        print(f'Abrindo detalhes para: {ultima_previsao}')
        abrir_janela_informacoes(ultima_previsao)
    else:
        messagebox.showwarning("Aviso", "Nenhuma previsão foi feita ainda. Aponte a câmera para um animal.")

def selecionar_pasta_e_treinar():
    pasta_imagens = filedialog.askdirectory()
    if pasta_imagens:
        treinar_modelo(pasta_imagens)
        habilitar_botao_analisar()

def habilitar_botao_analisar():
    if carregar_modelo():
        btn_analisar.config(state=tk.NORMAL)
    else:
        messagebox.showerror("Erro", "Falha ao carregar o modelo. Tente treinar primeiro.")

def interface_grafica():
    global btn_analisar
    root = tk.Tk()
    root.title("Reconhecimento de Animais")
    root.geometry("700x600")

    btn_treinar = tk.Button(root, text="Treinar Modelo", command=selecionar_pasta_e_treinar)
    btn_treinar.pack(pady=10)
    
    label_img = tk.Label(root)
    label_img.pack(pady=10)

    btn_camera = tk.Button(root, text="Iniciar Câmera", command=lambda: ativar_camera(label_img))
    btn_camera.pack(pady=5)

    btn_analisar = tk.Button(root, text="Ver Detalhes do Animal", state=tk.DISABLED, command=capturar_e_analisar)
    btn_analisar.pack(pady=5)
    
    habilitar_botao_analisar()

    root.mainloop()

    if cap:
        cap.release()

if __name__ == "__main__":
    interface_grafica()