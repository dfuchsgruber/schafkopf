import pytorch_lightning as pl
from data.rufspiel_lange_karte import RufspielLangeKarteDataModule
from model.agent import Agent

trainer = pl.Trainer()
model = Agent(embedding_dim=16)
data = RufspielLangeKarteDataModule('games/rufspiel_lange_karte')
trainer.fit(model, data)
