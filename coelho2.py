import numpy as np
import pygame

# Definir tamanho da janela
largura, altura = 800, 600
tela = pygame.display.set_mode((largura, altura))

def ler_OFF(arquivo):
    with open(arquivo, 'r') as f:
        # Ler a primeira linha do arquivo (deve ser 'OFF')
        assert f.readline().strip() == 'OFF'

        # Ler a segunda linha do arquivo (deve conter número de vértices e faces)
        num_vertices, num_faces, _ = map(int, f.readline().strip().split())

        # Ler os vértices
        pontos = []
        for i in range(num_vertices):
            ponto = list(map(float, f.readline().strip().split()))
            assert len(ponto) == 3
            pontos.append(ponto)

        # Ler as faces
        faces = []
        for i in range(num_faces):
            face = list(map(int, f.readline().strip().split()))
            num_indices = face[0]
            faces.append(face[1:])

        return np.array(pontos,dtype=float), faces

def desenhar_faces(pontos, faces):


    # Definir cor de fundo
    cor_fundo = (255, 255, 255)

    # Definir cor das faces
    cor_faces = (255, 0, 0)

    # Obter valores mínimo e máximo das coordenadas x, y e z dos pontos
    x_min, y_min, z_min = np.min(pontos, axis=0)
    x_max, y_max, z_max = np.max(pontos, axis=0)

    # Calcular fator de escala para ajustar as dimensões do modelo para caber na tela
    fator_escala = min(largura / (x_max - x_min), altura / (y_max - y_min), 1)

    # Desenhar cada face na tela
    for face in faces:
        pontos_face = pontos[face, :]
        pontos_face[:, 0] -= x_min
        pontos_face[:, 1] -= y_min
        pontos_face[:, :2] *= fator_escala
        pontos_face[:, 1] = altura - pontos_face[:, 1]
        pontos_face = pontos_face.astype(int)
        pontos_face = [(p[0],p[1]) for p in pontos_face]
        pygame.draw.polygon(tela, cor_faces, pontos_face)

    # Atualizar a tela


# Ler pontos, arestas e faces de um arquivo OFF
pontos, faces = ler_OFF('assets/bunny.off')
print("Acabei de ler o arquivo")

# Desenhar as faces na tela
# Loop para manter a janela aberta
pygame.init()
rodando = True
while rodando:
    desenhar_faces(pontos, faces)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            rodando = False
            pygame.quit()

    pygame.display.update()

