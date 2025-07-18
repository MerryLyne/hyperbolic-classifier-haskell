from gensim.models.poincare import PoincareModel

relations = [
    ('animal', 'mammal'),
    ('mammal', 'dog'),
    ('dog', 'beagle')
]

model = PoincareModel(train_data=relations, size=2, negative=2)
model.train(epochs=50)

print(model.kv.most_similar('dog'))
