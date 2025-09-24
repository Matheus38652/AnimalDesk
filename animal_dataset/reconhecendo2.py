import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk

model = None
class_indices = None
cap = None

DADOS_ANIMAIS = {
    "pinscher": {
        "vida": "15 anos",
        "altura": "30 CM",
        "peso": "6 KG",
        "destaques": "Cãezinhos dessa raça adoram correr por toda a casa, além de latir quando percebem algo diferente — ou seja, sempre!",
        "temperamento": "O temperamento do Pinscher vai variar do treinamento e da socialização. Geralmente eles costumam latir para estranhos ou em situações que consideram de risco, mas essa é uma característica de cão guardião. No fundo eles são amorosos e adoram brincar com os seus tutores.",
        "popularidade": "O cãozinho está entre as raças favoritas na Europa, Estados Unidos e aqui no Brasil também."
    },
    "bulldogue": {
        "vida": "8-10 anos",
        "altura": "31-40 cm",
        "peso": "18-23 kg",
        "destaques": "Conhecido por sua natureza dócil e calma.",
        "temperamento": "Amigável, corajoso e teimoso. Ótimo com crianças.",
        "popularidade": "Extremamente popular em todo o mundo."
    },
    "calopsita": {
        "vida": "10 a 15 anos",
        "altura": "Aproximadamente 32 cm",
        "peso": "80 a 120 g",
        "destaques": "É uma ave sociável e interativa, conhecida por seu charmoso topete que expressa seu humor. Pode aprender a assobiar melodias.",
        "temperamento": "Dócil, brincalhona e muito apegada aos tutores. Gosta de companhia e pode ficar barulhenta para chamar atenção, mas não costuma ser agressiva.",
        "popularidade": "É uma das aves de estimação mais populares no Brasil devido ao seu jeito carismático e interativo."
    },
    "cavalo": {
        "vida": "25 a 30 anos",
        "altura": "Varia de 1,55 a 1,70 m (na cernelha)",
        "peso": "500 a 600 kg",
        "destaques": "Animal herbívoro de grande porte, domesticado há milhares de anos para transporte, trabalho e esportes. Algumas raças podem atingir até 60 km/h.",
        "temperamento": "Nobre e dócil, mas pode se assustar facilmente. É um animal social que vive em grupos e se comunica muito pela postura corporal.",
        "popularidade": "Extremamente importante na história da humanidade, mantendo sua popularidade em esportes como hipismo e corridas."
    },
    "chimpanze": {
        "vida": "40 a 50 anos (na natureza)",
        "altura": "Até 1,5 m (em pé)",
        "peso": "40 a 70 kg (machos)",
        "destaques": "É o parente vivo mais próximo dos humanos. É conhecido pela sua impressionante inteligência e capacidade de usar ferramentas.",
        "temperamento": "Vive em grupos sociais complexos com hierarquias. É um animal comunicativo, que usa vocalizações e expressões faciais.",
        "popularidade": "Famoso na cultura popular e crucial em estudos científicos, mas é uma espécie ameaçada de extinção."
    },
    "pastor-alemao": {
        "vida": "9 a 13 anos",
        "altura": "60 a 65 cm (machos)",
        "peso": "30 a 40 kg (machos)",
        "destaques": "Raça extremamente versátil, famosa por atuar como cão de guarda, cão policial, em resgates e como guia.",
        "temperamento": "Muito leal, inteligente, protetor e obediente. É um cão confiante e corajoso, mas precisa de socialização e atividades físicas constantes.",
        "popularidade": "É uma das raças de cães mais populares e reconhecidas em todo o mundo."
    },
    "gato-persa": {
        "vida": "12 a 17 anos",
        "altura": "20 a 25 cm",
        "peso": "3 a 5,5 kg",
        "destaques": "Famoso por sua pelagem longa e exuberante, olhos grandes e expressivos, e um característico rosto achatado.",
        "temperamento": "É um gato calmo, dócil e afetuoso. Prefere ambientes tranquilos e não é muito propenso a pular e escalar como outras raças de gatos.",
        "popularidade": "Uma das raças de gatos mais antigas e conhecidas mundialmente, apreciada por sua beleza e temperamento tranquilo."
    }
}

def abrir_janela_informacoes(especie):
    chave_normalizada = especie.lower().replace(" ", "-").replace("_", "-")
    info = DADOS_ANIMAIS.get(chave_normalizada)

    if not info:
        messagebox.showinfo("Resultado", f'O animal é: {especie.capitalize()}')
        return

    janela_info = tk.Toplevel()
    janela_info.title(f"Informações sobre: {especie.capitalize()}")
    janela_info.geometry("400x550")
    
    lbl_titulo = tk.Label(janela_info, text=especie.capitalize(), font=("Arial", 16, "bold"))
    lbl_titulo.pack(pady=10)

    frame_stats = tk.Frame(janela_info)
    frame_stats.pack(pady=5, padx=10, fill="x")

    lbl_vida = tk.Label(frame_stats, text=f"Expectativa de vida: {info['vida']}", font=("Arial", 10))
    lbl_vida.pack(anchor="w")
    
    lbl_altura = tk.Label(frame_stats, text=f"Média de Altura: {info['altura']}", font=("Arial", 10))
    lbl_altura.pack(anchor="w")

    lbl_peso = tk.Label(frame_stats, text=f"Média de Peso: {info['peso']}", font=("Arial", 10))
    lbl_peso.pack(anchor="w")
    
    def criar_secao(parent, titulo, texto):
        lbl_titulo_secao = tk.Label(parent, text=titulo, font=("Arial", 12, "bold"))
        lbl_titulo_secao.pack(pady=(15, 2), anchor="w", padx=10)
        
        lbl_texto_secao = tk.Label(parent, text=texto, wraplength=380, justify="left", font=("Arial", 10))
        lbl_texto_secao.pack(anchor="w", padx=10)

    criar_secao(janela_info, "Destaques e Curiosidades", info['destaques'])
    criar_secao(janela_info, "Temperamento", info['temperamento'])
    criar_secao(janela_info, "Popularidade", info['popularidade'])


def verificar_estrutura_pasta(pasta_imagens):
    if not os.path.exists(pasta_imagens): return False
    subpastas = [f.path for f in os.scandir(pasta_imagens) if f.is_dir()]
    if len(subpastas) == 0: return False
    for subpasta in subpastas:
        if len([f for f in os.scandir(subpasta) if f.is_file()]) == 0: return False
    return True

def treinar_modelo(pasta_imagens):
    global model, class_indices
    if not verificar_estrutura_pasta(pasta_imagens):
        messagebox.showerror("Erro de Estrutura", "A pasta selecionada não contém subpastas com imagens. Verifique a estrutura.")
        return
    
    print("Iniciando o treinamento do modelo. Isso pode levar vários minutos...")
    messagebox.showinfo("Treinamento", "O treinamento foi iniciado. A janela pode parecer travada, por favor aguarde a mensagem de conclusão.")

    datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    train_generator = datagen.flow_from_directory(pasta_imagens, target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=20)
    
    model.save('meu_modelo.h5')
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
        
    messagebox.showinfo("Treinamento Concluído", "Modelo treinado e salvo com sucesso!")

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
    
    indices_para_classes = {v: k for k, v in class_indices.items()}
    nome_classe = indices_para_classes[classe_index]
    
    return nome_classe

def atualizar_imagem(label_img):
    global cap
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
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

def capturar_e_analisar(label_img):
    if cap is None or not cap.isOpened():
        messagebox.showwarning("Aviso", "Por favor, inicie a câmera primeiro.")
        return
        
    ret, frame = cap.read()
    if ret:
        especie_reconhecida = reconhecer_objeto(frame, class_indices)
        print(f'Previsão do modelo: {especie_reconhecida}')
        abrir_janela_informacoes(especie_reconhecida)
    else:
        messagebox.showerror("Erro", "Não foi possível capturar a imagem.")

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

    btn_analisar = tk.Button(root, text="Analisar Animal da Câmera", state=tk.DISABLED, command=lambda: capturar_e_analisar(label_img))
    btn_analisar.pack(pady=5)
    
    habilitar_botao_analisar()

    root.mainloop()

    if cap:
        cap.release()

if __name__ == "__main__":
    interface_grafica()