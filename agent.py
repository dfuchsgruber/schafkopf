import numpy as np
import torch
import torch.nn.functional as F

class QNet(torch.nn.Module):
    """ Embeds a sequence of tricks. """
    
    def __init__(self, emb_dim, output_dim, num_game_contexts=3, num_layers_trick_embedder=2, trick_embedder_bidirectional=True):
        super().__init__()
        self.num_layers_trick_embedder = num_layers_trick_embedder
        self.num_directions = 2 if trick_embedder_bidirectional else 1
        self.emb_dim = emb_dim

        self.card_player_embedding = torch.nn.Embedding(num_embeddings=32 + 4 + 1, embedding_dim=emb_dim, padding_idx=0)
        self.game_context_embedding = torch.nn.Embedding(num_embeddings=num_game_contexts, embedding_dim=output_dim)
        self.trick_embedder = torch.nn.GRU(9 * emb_dim, output_dim, num_layers=num_layers_trick_embedder, batch_first=True, bidirectional=trick_embedder_bidirectional)
        self.fc = torch.nn.Linear(self.num_directions * output_dim + emb_dim, output_dim)

    def forward(self, tricks, current_trick, game_context, actions):
        """ Forward pass through the trick embeddings. 
        
        Parameters:
        -----------
        tricks : torch.Tensor, shape [N, 8, 9]
            Tricks in the sequence. Dimension: 2 per card in a trick, corresponding to the player and the played card. Another one corresponding
            to the player who won that trick. Missing values are zero-padded.
        current_trick : int
            Index of the trick of the current state ( in [0, 8[ )
        game_context : torch.Tensor, shape [N]
            The game context (who plays which kind of game).
        actions : torch.Tensor, shape [N, A]
            All the cards the player currently has in hand.

        Returns:
        --------
        Q : torch.Tensor, shape [N, A]
            Q values for each each action.
        """
        # Embed the actions
        actions = self.card_player_embedding(actions) # shape [N, A, emb_dim]
        N, A, _ = actions.size()

        # Use the game context as initialization for the trick embedder
        game_context = self.game_context_embedding(game_context).unsqueeze(0) # shape [1, N, out_dim]
        game_context = game_context.expand((self.num_layers_trick_embedder * self.num_directions, N, -1)) # shape [N, n_dir, out_dim]

        # Embed tricks
        tricks = self.card_player_embedding(tricks) # shape [N, 8, 9, emb_dim]
        tricks = tricks.view((-1, 8, 9 * self.emb_dim)) # shape [N, 72, emb_dim]
        tricks, _ = self.trick_embedder(tricks, game_context) # shape [N, 8, n_dir * out_dim]
        tricks_embedding = tricks[:, current_trick, :] # shape [N, n_dir * out_dim]
        tricks_embedding = tricks_embedding.unsqueeze(1).expand((N, A, -1)) # shape [N, A, n_dir * output_dim]

        # Concatenate the trick embedding with the actions
        actions = torch.cat((actions, tricks_embedding), dim=-1)
        q = self.fc(actions)
        return q

