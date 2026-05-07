import pygame
import numpy as np
import torch
class DrawScreen:

    def __init__(self,w,h,pixel):
        pygame.init()
        self.screen = pygame.display.set_mode((w,h))
        self.clock = pygame.time.Clock()
        self.board = np.full((pixel,pixel),0)
        self.pixel = pixel
        self.pixelSize = int(w/pixel)
        self.running = True
        self.centerstrokestrength = 1
        self.sidestrokestrength = 0.5
        self.colorval = 255
        self.font = pygame.font.SysFont("Arial", 36)
        self.string = ""
        self.text = self.font.render("Hello Pygame!", True, (255, 255, 255))
    
    def display(self):
        for i in range(self.pixel):
            for j in range(self.pixel):
                val = self.board[i][j]
                c = (val,val,val)
                x = j*self.pixelSize
                y = i*self.pixelSize
                pygame.draw.rect(self.screen,c,pygame.Rect(x,y,self.pixelSize,self.pixelSize))
    
    def setMessage(self,str):
        self.text = self.font.render(str, True, (255,255,255))

    def input(self):
        self.colorval = 255
        if pygame.mouse.get_pressed()[0] == True or pygame.mouse.get_pressed()[2] == True:
            x = pygame.mouse.get_pos()[0]
            y = pygame.mouse.get_pos()[1]

            i = int(y / self.pixelSize)
            j = int(x / self.pixelSize)

            if pygame.mouse.get_pressed()[0] == True:
                self.colorval = 255
            elif pygame.mouse.get_pressed()[2] == True:
                self.colorval = 0

            if( i >=  0 and i < self.pixel and j >= 0 and j< self.pixel):
                self.board[i][j] = self.colorval 
                if ( i >= 1):
                    self.board[i-1][j] += self.colorval * self.sidestrokestrength
                    self.board[i-1][j] = min(self.board[i-1][j],self.colorval)
                if ( i < self.pixel-1):
                    self.board[i+1][j] += self.colorval * self.sidestrokestrength
                    self.board[i+1][j] = min(self.board[i+1][j],self.colorval)
                if ( j >= 1):
                    self.board[i][j-1] += self.colorval * self.sidestrokestrength
                    self.board[i][j-1] = min(self.board[i][j-1],self.colorval)
                if ( j < self.pixel-1):
                    self.board[i][j+1] += self.colorval * self.sidestrokestrength
                    self.board[i][j+1] = min(self.board[i][j+1],self.colorval)
            
    def run(self,model = None,categories = None):
        self.setMessage("hello")
        time = pygame.time.get_ticks()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            if (not model == None) and (pygame.time.get_ticks() - time > 1000):
                data = torch.tensor(self.board,dtype=torch.float32).reshape(-1,1,28,28) / 255
                out = model(data)
                preds = torch.argmax(out, dim=1).item()
                self.setMessage(f"I predict that it is a {categories[preds]}")
                time = pygame.time.get_ticks()
            self.input()
            self.display()
            self.screen.blit(self.text,(0,0))
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
