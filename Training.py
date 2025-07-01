from EntireModel import TrDCN
from UtilityFunctions import *
from Hyperparameters import parameters

# 本文件定义训练器

class Trainer:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.trainer = parameters['optimizer'](self.model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
        self.scheduler = parameters['scheduler'](optimizer=self.trainer,T_max=parameters['epochs'])

    def train(self):
        total_losses, word_losses, counting_losses = [], [], []

        for epoch in range(parameters['epochs']):
            l1, l2, l3 = [], [], []
            
            for i, (images, masks, labels, lengths) in enumerate(self.dataloader):
                if i >= parameters['max_batches']:
                    break

                self.model.train()
                self.trainer.zero_grad()

                word_probs, count_vector, word_loss, counting_loss, attention_weights = self.model(images, masks, labels, lengths)
        
                loss = word_loss + parameters['lamda'] * counting_loss
                print(f"epoch {epoch + 1}, batch {i + 1}, loss: {loss}")
                
                l1.append(loss.detach().cpu().item())
                l2.append(word_loss.detach().cpu().item())
                l3.append(counting_loss.detach().cpu().item())
                
                # 优化器再每个批量结束后立即调用
                loss.backward()
                self.trainer.step()

            # CosineAnnealingLR的学习率调度器按训练周期调整学习率
            self.scheduler.step()
            
            total_losses.append(np.mean(l1))
            word_losses.append(np.mean(l2))
            counting_losses.append(np.mean(l3))
        
        # 绘制所示数值曲线
        plt.plot(total_losses, label='Total Loss',  color='red')
        plt.plot(word_losses, label='Word Loss', color='blue')
        plt.plot(counting_losses, label='Counting Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
        print(f"Training finished!")    
