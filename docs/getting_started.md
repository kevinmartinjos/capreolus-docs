# Basic usage


```python
import capreolus

config = {
	"reranker": "KNRM"
}
capreolus.train(config)

```

The `train()` method expects a config dict as an argument, which at least specifies which [reranker](brokenlink) to use. Capreolus plugs in sane defaults for every other possible configuration - for example, unless otherwise specified, capreolus will a dummy benchmark for training. This is probably not what you want.


# Creating a config and training

You can a pass a configuration dict into capreolus to get it to do specific things. The config dict also holds the hyperparameters that might be required by your neural model to train. For example, let us train a KNRM model on the [Robust04](brokenlink) dataset. We do not care about the accuracy of the model - we just want to see if it works - hence it makes sense to train it on just 10 iterations:

```python
import capreolus
config = {
	"reranker": "KNRM",
	"benchmark": "rob04.title",
	"niters": 10
}

trained_model = capreolus.train(config)
```

See the [datasets](brokenlink) page for a list of supported datasets

The set of supported config parameters vary between rerankers. See the documentation page of the specific [reranker](brokenlink). Alternatively, all parameters that "go" with a base configuration can be obtained using `get_supported_params()`:

```python
import capreolus

config = {
	"reranker": "PACRR"
}
print(capreolus.get_supported_params(config))

config = {
	"reranker": "PACRR",
	"benchmark": "rob04.title",
	"index": "terrier"
}

# We now see index params supported only by terrier. If an index is not specified, get_supported_params() assumes that anserini is the indexing engine and shows you parameters specific to that
print(capreolus.get_supported_params(config))

```

In every reranking task, the first step is to retrieve a set of documents using a traditional IR method, like BM25 (see [how capreolus works](brokenlink)). Unless otherwise specified, the initial "searcher" component is BM25. However this can changed to, say, BM25 with query expansion, through the config dict:


```python
config = {
	"reranker": "PACRR",
	"benchmark": "rob04.title",
	"index": "terrier",
	"searcher": "bm25rm3yang19"
}

# Now displays BM25RM3 specific config params
print(capreolus.get_supported_params(config))
```


# Evaluating a trained model

```python

trained_model = capreolus.train(config)

# results will be a dict of the form: {'map': 0.3423, 'ndcg': '0.12312', ...}
results = capreolus.evaluate(trained_model)
```

Unless otherwise specified, `evaluate()` will calculate all supported metrics - precision, mean average precision (MAP), NDCG, and NDCG and MAP with different cuts. See the documentation on [evaluation](brokenlink) for an overview of supported metrics

It is also possible to train a reranker on dataset A and then evaluate it on dataset B:

```python
results = capreolus.evaluate(trained_model, benchmark='clueweb12')
```

# Custom datasets

It is possible to use your own data to train and evaluate rerankers, as long as they are in [TREC format](brokenlink):

```python
import capreolus

config = {
	"reranker": "KNRM",
	"benchmark": "trec",
	"collection.qrels": "path to qrels",
	"collection.topics": "path to topics",
	"collection.data": "path to documents",
	"collection.folds": "path to json file that lays down the folds",
}

trained_model = capreolus.train(config)
```

See the [custom datasets](brokenlink) page for more information on how to convert your data into a format compatible with capreolus

# Custom reranker

You can ask capreolus to train and evaluate on your custom reranker. Here's an example that uses a very barebones FCC:

```python
import capreolus
from torch import nn

class FCC(nn.Module):
	def __init__(self, embeddings, config):
		"""
		embeddings - Usually a torch tensor used to lookup up vectors corresponding to words in vocabolary. See "create_embeddings_matrix()" in <link to extractors>
		config - the same config that you pass to train(), but updated with default values for all the configs that you did not explicitly specify
		"""
		super(KNRM_class, self).__init__()
		self.layers = nn.Sequential(*[nn.Linear(300, 400), nn.Linear(400, 50), nn.Tanh()])

	def forward(self, doctoks, querytoks):
		x = "a vector that you created based on doctoks and query toks"
		return self.layers(x)

class MyReranker(capreolus.reranker.Reranker):
	# Should be one of the supported extractors. 
	extractor = capreolus.extractor.BagOfWords

	def __init__(self, *args, **args):
		pass

	def config(self):
		"""
		The set of hyperparameters that your reranker expects and their default values 
		"""
		pass

	def score(self, query, posdoc, negdoc):
		"""
		query, posdoc, and negdoc could be vectors, depending on the extractor that you have chosen
		"""
		return [
			self.model(query, posdoc),
			self.model(query, negdoc)
		]

	def test(self, query, doc):
		return self.model(query, doc)
```

In order to make capreolus your custom model, all you have to do is pass the concrete class instead of the usual string in the config:

```python
config = {
	"reranker": MyReranker,
	"benchmark": "rob04.title",
	"niters": 100
}

trained_model = capreolus.train(config)
```

For more details on the exact interface that your custom reranker should implement, refer to the [rerankers](brokenlink) page


# Custom Extractor

Coming soon ;)