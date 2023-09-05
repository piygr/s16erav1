# Session 16 Assignment
Building a full transformer for language translation 
### Instructions
- Pick the "en-fr" dataset from opus_books
- Remove all English sentences with more than 150 "tokens"
- Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
- Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8
## Transformer Architecture
<img width="1013" alt="transformers" src="https://github.com/piygr/s15erav1/assets/135162847/610de7d6-d869-4841-bb79-ee43ba1a692e">


------
## models/TransformerV2Lightning.py
The file contains **TransformerV2LightningModel** - a Transformer model written in pytorch-lightning as desired in the assignment. 

Here is the summary of the network -

```
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | Encoder            | 6.3 M 
1 | decoder          | Decoder            | 9.4 M 
2 | projection_layer | ProjectionLayer    | 15.4 M
3 | src_embed        | InputEmbeddings    | 15.4 M
4 | tgt_embed        | InputEmbeddings    | 15.4 M
5 | src_pos          | PositionalEncoding | 0     
6 | tgt_pos          | PositionalEncoding | 0     
7 | loss_fn          | CrossEntropyLoss   | 0     
--------------------------------------------------------
61.8 M    Trainable params
0         Non-trainable params
61.8 M    Total params
247.392   Total estimated model params size (MB)
```

## dataset.py
opus_books - HuggingFace dataset & tokenizer is used as raw dataset. The translation language pair is en (English) to fr (French).

## S16.ipynb
The file is an IPython notebook. This was run separately in Kaggle to train the model.


```
Epoch  0
Train Loss: 5.837872
------
SOURCE := [The reader will remember that these men were mixed up in the secret politics of Louis XI.]
EXPECTED := ['On se souvient que ces deux hommes étaient mêlés à la politique secrète de Louis XI.']
PREDICTED := ['Le monde me ces hommes étaient dans le roi .']
Validation cer: 0.5714285969734192
Validation wer: 0.8125
Validation BLEU: 0.0
--------------------
Epoch  1
Train Loss: 4.447019
------
SOURCE := [Monsieur Porthos!" cried the procurator’s wife. "I have been wrong; I see it. I ought not to have driven a bargain when it was to equip a cavalier like you."]
EXPECTED := ["«Monsieur Porthos! monsieur Porthos! s'écria la procureuse, j'ai tort, je le reconnais, je n'aurais pas dû marchander quand il s'agissait d'équiper un cavalier comme vous!»"]
PREDICTED := ["-- Monsieur Porthos ! s ' écria la procureuse ; je suis bien sûr ; je ne sais pas , je ne serais pas obligé de m ' avoir été quand il était à un cavalier comme vous ."]
Validation cer: 0.5058139562606812
Validation wer: 1.2000000476837158
Validation BLEU: 0.0
--------------------
Epoch  2
Train Loss: 3.807128
------
SOURCE := ["I was privately married, and I retired from the stage.]
EXPECTED := ["-- Je me suis mariée secrètement et j'ai quitté le théâtre."]
PREDICTED := ["-- J ' ai été marié , et je me mis à la scène ."]
Validation cer: 0.6271186470985413
Validation wer: 1.1818181276321411
Validation BLEU: 0.0
--------------------
Epoch  3
Train Loss: 3.462492
------
SOURCE := [It would not have been wise to tell how we came there. The superstitious Italians would have set us down for fire-devils vomited out of hell; so we presented ourselves in the humble guise of shipwrecked mariners.]
EXPECTED := ["Dire comment nous étions arrivés dans l'île ne nous parut pas prudent: l'esprit superstitieux des Italiens n'eût pas manqué de voir en nous dés démons vomis du sein des enfers; il fallut donc, se résigner à passer pour d'humbles naufragés."]
PREDICTED := ["Ce n ' eût pas été sage de dire à quel moment nous arrivâmes ; le nous aurait mis à tirer pour tirer pour feu des diables d ' enfer ; si nous nous nous en somme de naufragés ."]
Validation cer: 0.6485355496406555
Validation wer: 1.0
Validation BLEU: 0.0
--------------------
Epoch  4
Train Loss: 3.251329
------
SOURCE := ["We had been a fortnight at the bottom of a hole undermining the railway, and it was not the imperial train that was blown up, it was a passenger train.]
EXPECTED := ["Nous étions restés quatorze jours au fond d'un trou, a miner la voie du chemin de fer; et ce n'est pas le train impérial, c'est un train de voyageurs qui a sauté…"]
PREDICTED := ["-- Nous avions quinze jours au fond d ' un trou , et ce n ' était pas le train de qui fut tiré , c ' était un train de voyage ."]
Validation cer: 0.5493826866149902
Validation wer: 0.90625
Validation BLEU: 0.0
--------------------
Epoch  5
Train Loss: 3.108443
------
SOURCE := [During all this time I can only once remember that there was the slightest disagreement between him and my mother.]
EXPECTED := ["Je ne me souviens pas qu'il y ait eu le moindre désaccord entre lui et ma mère, excepté une fois."]
PREDICTED := ["Pendant tout ce temps , je ne puis me rappeler qu ' une fois , c ' était le moindre cadence entre lui et ma mère ."]
Validation cer: 0.8041236996650696
Validation wer: 1.0499999523162842
Validation BLEU: 0.0
--------------------
Epoch  6
Train Loss: 2.987513
------
SOURCE := ["Ugh!" cried Gringoire, "what a great king is here!"]
EXPECTED := ['– Ouf ! s’écria Gringoire, que voilà un grand roi ! »']
PREDICTED := ['« Hum ! cria Gringoire , qu ’ est là - bas ! »']
Validation cer: 0.43396225571632385
Validation wer: 0.9166666865348816
Validation BLEU: 0.0
--------------------
Epoch  7
Train Loss: 2.877975
------
SOURCE := [Adieu, ma Clélia, je bénis ma mort puisqu’elle a été l’occasion de mon bonheur.]
EXPECTED := ['Farewell, my Clelia, I bless my death since it has been the cause of my happiness."']
PREDICTED := ['Good - bye , my Signora , I shall have my death since she was an opportunity of my happiness .']
Validation cer: 0.5542168617248535
Validation wer: 0.9375
Validation BLEU: 0.0
--------------------
Epoch  8
Train Loss: 2.706892
------
SOURCE := [I have myself set an example by making a match with Sir Lothian Hume, the terms of which will be communicated to you by that gentleman."]
EXPECTED := ["J'en ai moi-même donné l'exemple en faisant avec Sir Lothian Hume un match dont les conditions vont vous être communiquées par ce gentleman."]
PREDICTED := ['Je me suis mis au exemple en faisant un mariage avec Sir Lothian Hume , dont les termes vous seront par ce gentleman .']
Validation cer: 0.4714285731315613
Validation wer: 0.695652186870575
Validation BLEU: 0.0
--------------------
Epoch  9
Train Loss: 2.531077
------
SOURCE := [One of them was suddenly shut off.]
EXPECTED := ['L’une d’elles s’effaça brusquement.']
PREDICTED := ['L ’ un d ’ eux fut brusquement close .']
Validation cer: 0.6571428775787354
Validation wer: 2.5
Validation BLEU: 0.0
--------------------
Epoch  10
Train Loss: 2.371687
------
SOURCE := [But she managed only to exasperate Meaulnes.]
EXPECTED := ['Mais elle ne fit qu’exaspérer Meaulnes.']
PREDICTED := ['Mais elle ne réussit qu ’ à exaspérer Meaulnes .']
Validation cer: 0.25641027092933655
Validation wer: 1.1666666269302368
Validation BLEU: 0.0
--------------------
Epoch  11
Train Loss: 2.238340
------
SOURCE := ["A strange wish, Mrs. Reed; why do you hate her so?"]
EXPECTED := ['-- Étrange désir, madame Reed! Pourquoi la haïssiez-vous?']
PREDICTED := ['-- Un étrange désir , madame Reed ; pourquoi la - vous ainsi ?']
Validation cer: 0.3684210479259491
Validation wer: 1.375
Validation BLEU: 0.0
--------------------
Epoch  12
Train Loss: 2.115552
------
SOURCE := [I've the eye of an American!"]
EXPECTED := ['J’ai l’oeil américain.']
PREDICTED := ["J ' ai l ' oeil d ' un Américain ?"]
Validation cer: 0.7272727489471436
Validation wer: 3.6666667461395264
Validation BLEU: 0.0
--------------------
Epoch  13
Train Loss: 2.011159
------
SOURCE := ["Oh, I cannot believe you!"]
EXPECTED := ['-- Oh! je ne puis vous croire!']
PREDICTED := ['-- Oh ! je ne puis vous croire !']
Validation cer: 0.06666667014360428
Validation wer: 0.5714285969734192
Validation BLEU: 0.0
--------------------
Epoch  14
Train Loss: 1.926809
------
SOURCE := [A single man of large fortune; four or five thousand a year.]
EXPECTED := ['Quatre ou cinq mille livres de rente !']
PREDICTED := ['Un seul homme de grande fortune , quatre ou cinq mille ans de rentes .']
Validation cer: 1.105263113975525
Validation wer: 1.375
Validation BLEU: 0.0
--------------------
Epoch  15
Train Loss: 1.871439
------
SOURCE := ["Oh, yes; I am," added the king, taking a handful of gold from La Chesnaye, and putting it into the hand of d’Artagnan.]
EXPECTED := ["-- Oui, je le suis, ajouta le roi en prenant une poignée d'or de la main de La Chesnaye, et la mettant dans celle de d'Artagnan."]
PREDICTED := ["-- Oh ! oui , je suis , ajouta le roi en prenant une poignée d ' or de La Chesnaye , et qu ' il l ' a remise à d ' Artagnan ."]
Validation cer: 0.3515625
Validation wer: 0.9230769276618958
Validation BLEU: 0.0
--------------------
Epoch  16
Train Loss: 1.849386
------
SOURCE := [Its unstable color would change with tremendous speed as the animal grew irritated, passing successively from bluish gray to reddish brown.]
EXPECTED := ["Sa couleur inconstante, changeant avec une extrême rapidité suivant l'irritation de l'animal, passait successivement du gris livide au brun rougeâtre."]
PREDICTED := ['Cette rapidité devait être avec une rapidité extrême , car le fauve , passant successivement par le doux bruissement des eaux rougeâtres .']
Validation cer: 0.5933333039283752
Validation wer: 0.949999988079071
Validation BLEU: 0.0
--------------------
Epoch  17
Train Loss: 1.830442
------
SOURCE := [After a long interval, when they were able to speak:]
EXPECTED := ['Bien longtemps après, quand on put parler :']
PREDICTED := ['Après un long moment , quand on put parler :']
Validation cer: 0.39534884691238403
Validation wer: 0.625
Validation BLEU: 0.0
--------------------
Epoch  18
Train Loss: 1.815035
------
SOURCE := [His face became convulsed, his limbs rigid, his nerves could be seen knotting beneath his skin.]
EXPECTED := ['Sa face se convulsionnait, ses membres se raidissaient; on voyait que les nerfs se nouaient en lui.']
PREDICTED := ['Il avait la figure , les membres raidis , le visage contenu , il voyait sous sa peau .']
Validation cer: 0.6565656661987305
Validation wer: 1.058823585510254
Validation BLEU: 0.0
--------------------
Epoch  19
Train Loss: 1.801223
------
SOURCE := [He had lived at court and slept in the bed of queens!]
EXPECTED := ['Il avait vécu à la Cour et couché dans le lit des reines!']
PREDICTED := ['Et il avait vécu en cour , il dormait au lit !']
Validation cer: 0.5789473652839661
Validation wer: 0.8461538553237915
Validation BLEU: 0.0
--------------------
Epoch  20
Train Loss: 1.789502
------
SOURCE := ["However," observed Cyrus Harding, "here we are in an impregnable position.]
EXPECTED := ['«Toutefois, fit observer Cyrus Smith, nous sommes ici dans une situation inexpugnable.']
PREDICTED := ['« Cependant , fit observer Cyrus Smith , nous sommes ici dans une position .']
Validation cer: 0.3255814015865326
Validation wer: 0.5833333134651184
Validation BLEU: 0.0
--------------------
Epoch  21
Train Loss: 1.778268
------
SOURCE := ["Oh! that's where it is, is it?" replied the man; "well, you take my advice and go there quietly, and take that watch of yours with you; and don't let's have any more of it."]
EXPECTED := ['« Ah ! c’est la que vous logez, dites-vous ? reprit l’homme. Eh bien, je vous conseille de remettre votre montre dans votre poche et de rentrer chez vous tranquillement.']
PREDICTED := ["-- Oh ! c ' est là , c ' est cela ? répondit l ' homme . Eh bien , prenez mon conseil et y allez tranquillement , prenez cette montre de la vôtre à votre égard , et ne vous en inquiétez pas davantage ."]
Validation cer: 0.6804733872413635
Validation wer: 1.3666666746139526
Validation BLEU: 0.0
--------------------
Epoch  22
Train Loss: 1.770308
------
SOURCE := [I was at school at the time, and the adventure appeared to me to be cruel for the king."]
EXPECTED := ["J'étais au séminaire à cette époque, et l'aventure me parut cruelle pour le roi."]
PREDICTED := ["J ' étais à l ' école au moment où l ' aventure me parut cruelle pour le roi ."]
Validation cer: 0.375
Validation wer: 1.0
Validation BLEU: 0.0
--------------------
Epoch  23
Train Loss: 1.762439
------
SOURCE := [By the date of the tests, April, 189-I realized that Meaulnes had started it only a few days before leaving Sainte-Agathe.]
EXPECTED := ['À la date des devoirs, avril 1892… je reconnus que Meaulnes l’avait commencé peu de jours avant de quitter Sainte-Agathe.']
PREDICTED := ['À la date du , avril , je compris que Meaulnes était parti depuis quelques jours seulement avant de quitter Sainte - Agathe .']
Validation cer: 0.44628098607063293
Validation wer: 0.6499999761581421
Validation BLEU: 0.0
--------------------
Epoch  24
Train Loss: 1.757182
------
SOURCE := [And the innkeeper, who was very excited, talked more freely, repeating that he only asked possibilities from the masters, without demanding, like so many others, things that were too hard to get.]
EXPECTED := ["Et le cabaretier, tres excité, se livra davantage, tout en répétant qu'il demandait seulement le possible aux patrons, sans exiger, comme tant d'autres, des choses trop dures a obtenir."]
PREDICTED := ["Et le cabaretier , qui était tres ému , parlait plus longuement , répétait qu ' il ne demandait rien a des maîtres sans demander , comme des autres choses , trop difficiles a trouver ."]
Validation cer: 0.5351351499557495
Validation wer: 1.0
Validation BLEU: 0.0
--------------------
```

## How to setup locally
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
