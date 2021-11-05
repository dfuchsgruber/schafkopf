import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Agent(pl.LightningModule):
    """ Basic agent for a Rufspiel. """

    def __init__(self, 
        num_actions: int = 32, 
        embedding_dim: int = 64,
        ):

        self.card_player_embedding = nn.Embedding(32 + 4, embedding_dim, padding_idx=0)
        
        self.color_embedding = nn.Embedding(4, embedding_dim)
        
        self.trick_embedding = nn.ModuleList([
            nn.Linear(9 * embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        ])

        self.game_context_embedding = nn.ModuleList([
            nn.Linear(2 * embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        ])

        self.hand_cards_embedding = nn.ModuleList([
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
        ])

        self.head = nn.ModuleList([
            nn.Linear(4 * embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        ])

    def forward(self, batch):

        # The game state is the color and the player
        color = self.color_embedding(batch['color']) # batch x h
        game_player = self.card_player_embedding(batch['player']) # batch x h
        game_context = torch.cat([color, game_player], dim=-1) # batch x 2h
        game_context = self.game_context_embedding(game_context)

        hand_cards = self.card_player_embedding(batch['agent_cards']).max(dim=-2) # batch x h

        tricks = self.card_player_embedding(batch['state'][:, :, :8]) # batch x 8 x 8 x h
        tricks = self.trick_embedding(tricks.view((tricks.size(0), tricks.size(1), -1)) # batch x 8 x h
        trick_winner = self.card_player_embedding(batch['state'][:, :, 8]) # batch x 8 x h
        tricks = torch.cat([tricks, trick_winner], dim=-2) # batch x 8 x 2h
        tricks = tricks.max(dim=-2) # batch x 2h

        
        state = torch.cat([game_context, hand_cards, tricks,], dim=-1) # batch x 4h
        return self.head(state)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, batch['action'])
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, batch['action'])
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


