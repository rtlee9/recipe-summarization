# Recipe summarization

This repo implements a sequence-to-sequence encoder-decoder using Keras to summarize recipe instructions by predicting a recipe title. This code is based on Siraj Raval's [How to Make a Text Summarizer](https://github.com/llSourcell/How_to_make_a_text_summarizer); it won the coding challenge for that week's [video](https://www.youtube.com/watch?v=ogrJaOIuBx4), and was featured in the following week's [video](https://www.youtube.com/watch?v=nRBnh4qbPHI).

![youtube](youtube_screenshot.jpg)

This repo has been updated since then, so please check out tag `v1.0.0` to view the version associated with the coding challenge. Lastly, note that this repo is not being actively maintained -- I will do my best to respond to any issues opened but make no guarantees.

Please consider buying me a coffee if you like my work:

<a href="https://www.buymeacoffee.com/6Ii7vzL" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;" ></a>

## Data
I scraped 125,000 recipes from various websites for training (additional details can be found [here](https://github.com/rtlee9/recipe-box)). Each recipe consists of:

* A recipe title
* A list of ingredients
* Preparation instructions
* An image of the prepared recipe (missing for ~40% of recipes collected)

The model was fitted on the recipe ingredients, instructions and title. Ingredients were concatenated in their original order to the instructions. Recipe images were not used for this model.

## Training
This model was trained for ~6 hours on an nVidia Tesla K80. Training consisted of several training iterations, in which I successively decremented the learning rate and incremented the ratio of flip augmentations.

## Sampled outputs
Below are a few _cherry-picked_ in-sample predictions from the model:

### Example 1:
* __Generated:__ Chicken Cake
* __Original:__ Chicken French - Rochester , NY Style
* __Recipe:__ all purpose flour ; salt ; eggs ; white sugar ; grated parmesan cheese ; olive oil ; skinless ; butter ; minced garlic ; dry sherry ; lemon juice ; low sodium chicken base ; ;Mix together the flour , salt , and pepper in a shallow bowl . In another bowl , whisk beaten eggs , sugar , and Parmesan cheese until the mixture is thoroughly blended and the sugar has dissolved . Heat olive oil in a large skillet over medium heat until the oil shimmers . Dip the chicken breasts into the flour mixture , then into the egg mixture , and gently lay them into the skillet . Pan-fry the chicken breasts until golden brown and no longer pink in the middle , about 6 minutes on each side . Remove from the skillet and set aside . In the same skillet over medium-low heat , melt the butter , and stir in garlic , sherry , lemon juice , and chicken base ...

### Example 2:
* __Generated:__ Fruit Soup
* __Original:__ Red Apple Milkshake
* __Recipe:__ red apple peeled ; cold skim milk ; white sugar ; fresh mint leaves for garnish ; ;In a blender , blend the apple , skim milk , and sugar until smooth . Garnish with mint to serve .

### Example 3:
* __Generated:__ Asparagus with Chicken
* __Original:__ Asparagus and Dill Avgolemono Soup
* __Recipe:__ asparagus ; chicken stock ; unsalted butter ; leek ; onion ; ribs celery ; salt ; water ; eggs ; juice of 2 lemons ; minced fresh dill ; dill sprigs for garnish ;Trim off ends of asparagus and using a vegetable peeler remove about 3 to 4-inches of the skin of each stalk , reserving both the ends and peels . Cut asparagus into 1-inch pieces , reserving tips for garnish . In a saucepan combine the asparagus peels and trimmings with the chicken stock , bring to a boil , remove from heat and allow stock to infuse for 15 minutes . Strain stock and reserve . In a pot of salted boiling water blanch the asparagus tips for 2 to 3 minutes , or until brilliant green and barely tender , and then refresh in a bowl of ice water . When tips are chilled , drain and reserve . In a large heavy pot melt the butter over moderate heat and cook the leeks , onion and celery , seasoned with salt and pepper , until softened , about 5 to 8 minutes . Add the 1-inch asparagus pieces and stir to combine ...

## Usage (Python 3.6)

* Clone repo: `git clone https://github.com/rtlee9/recipe-summarization.git && cd recipe-summarization`
* Initialize submodules: `git submodule update --init --recursive`
* Install dependencies [optional: in virtualenv]: `pip install -r requirements.txt`
* Setup directories: `python src/config.py`
* Download recipes from my Google Cloud Bucket: `wget -P recipe-box/data https://storage.googleapis.com/recipe-box/recipes_raw.zip; unzip recipe-box/data/recipes_raw.zip -d recipe-box/data` (alternatively, see the recipe-box submodule to scrape fresh recipe data)
* Tokenize data: `python src/tokenize_recipes.py`
* Initialize word embeddings with GloVe vectors:
  * Get GloVe vectors: `wget -P data http://nlp.stanford.edu/data/glove.6B.zip; unzip data/glove.6B.zip -d data`
  * Initialize embeddings: `python src/vocabulary-embedding.py`
* Train model: `python src/train_seq2seq.py`
* Make predictions: use src/predict.ipynb

## Next steps
Aside from tuning hyperparameters, there are a number of ways to potentially improve this model:

* Incorporate ingredients list non-sequentially, and add recipe images (see [recipe-box](https://github.com/rtlee9/recipe-box))
* Try different RNN sequence lengths, or [variable sequence lengths](https://danijar.com/variable-sequence-lengths-in-tensorflow/)
* Try different vocabulary sizes
