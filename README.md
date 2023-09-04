# Session 15 Assignment
To build a full transformer for language translation (Only to run for 10 epochs of training)

## Transformer Architecture
<img width="1013" alt="transformers" src="https://github.com/piygr/s15erav1/assets/135162847/610de7d6-d869-4841-bb79-ee43ba1a692e">


------
## models/TransformerV1Lightning.py
The file contains **TransformerV1LightningModel** - a Transformer model written in pytorch-lightning as desired in the assignment. 

Here is the summary of the network -

```
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | Encoder            | 12.6 M
1 | decoder          | Decoder            | 18.9 M
2 | projection_layer | ProjectionLayer    | 11.5 M
3 | src_embed        | InputEmbeddings    | 8.0 M 
4 | tgt_embed        | InputEmbeddings    | 11.5 M
5 | src_pos          | PositionalEncoding | 0     
6 | tgt_pos          | PositionalEncoding | 0     
7 | loss_fn          | CrossEntropyLoss   | 0     
--------------------------------------------------------
62.5 M    Trainable params
0         Non-trainable params
62.5 M    Total params
250.151   Total estimated model params size (MB)
```

## dataset.py
opus_books - HuggingFace dataset & tokenizer is used as raw dataset. The translation language pair is en (English) to it (Italian).

## S15.ipynb
The file is an IPython notebook. This was run separately in Kaggle to train the model.


```
Epoch  0
Train Loss: 6.412225
Validation Loss: 5.671622
------
SOURCE := [The next morning we would read that it was going to be a "warm, fine to set-fair day; much heat;" and we would dress ourselves in flimsy things, and go out, and, half-an-hour after we had started, it would commence to rain hard, and a bitterly cold wind would spring up, and both would keep on steadily for the whole day, and we would come home with colds and rheumatism all over us, and go to bed.]
EXPECTED := ['La mattina appresso leggemmo che sarebbe stata una «bella, calda, giornata». Ci vestimmo con gli abiti leggeri, e uscimmo, e mezz’ora dopo che eravamo partiti, si scatenò una fortissima pioggia, e si mise a imperversare un vento terribilmente freddo che durò tutto il giorno.']
PREDICTED := ['E il suo momento , che si a un ’ altro che la mia , e , e , e , e , e , e , e , e , e , e , e , e , e , e , e .']
Validation cer: 0.7381818294525146
Validation wer: 0.9545454382896423
Validation BLEU: 0.0
--------------------
Epoch  1
Train Loss: 5.675857
Validation Loss: 5.255237
------
SOURCE := [When he had done, instead of feeling better, calmer, more enlightened by his discourse, I experienced an inexpressible sadness; for it seemed to me--I know not whether equally so to others--that the eloquence to which I had been listening had sprung from a depth where lay turbid dregs of disappointment--where moved troubling impulses of insatiate yearnings and disquieting aspirations.]
EXPECTED := ['Quando egli ebbe terminato, invece di sentirmi più calma, più illuminata, provai una grande tristezza, perché mi pareva che quella eloquenza sgorgasse da una sorgente avvelenata da amare delusioni, e nella quale si agitavano desiderii insoddisfatti e aspirazioni angosciose.']
PREDICTED := ['Quando era stato , come se mi aveva fatto , e mi , e mi , e mi , e mi , e mi , e mi , e non mi , e mi , e mi .']
Validation cer: 0.7372262477874756
Validation wer: 0.9230769276618958
Validation BLEU: 0.0
--------------------
Epoch  2
Train Loss: 5.297083
Validation Loss: 4.992411
------
SOURCE := [You think all existence lapses in as quiet a flow as that in which your youth has hitherto slid away.]
EXPECTED := ['"Voi credete che tutta la vita sia calma come la vostra giovinezza.']
PREDICTED := ['Tu , come vi , come un uomo che ha fatto un uomo che ha fatto la vostra opinione .']
Validation cer: 0.7761194109916687
Validation wer: 1.4166666269302368
Validation BLEU: 0.0
--------------------
Epoch  3
Train Loss: 4.993914
Validation Loss: 4.831674
------
SOURCE := [His natural feelings prompted him to justify himself and prove that she was in the wrong; but to prove her in the wrong would mean irritating her still more, and widening the breach which was the cause of all the trouble.]
EXPECTED := ['Un sentimento istintivo pretendeva la giustificazione e la dimostrazione della colpa di lei; ma mostrare la colpa di lei significava irritarla maggiormente e aumentare quel distacco che era la causa di tutta la pena.']
PREDICTED := ['La sua vita era sempre sempre più di lui e che la sua situazione era stata in lui ; ma la sua cosa si era sempre più forte , e che la sua vita era stata sempre più di lui .']
Validation cer: 0.6527777910232544
Validation wer: 1.058823585510254
Validation BLEU: 0.0
--------------------
Epoch  4
Train Loss: 4.728517
Validation Loss: 4.722586
------
SOURCE := [He bowed, still not taking his eyes from the group of the dog and child.]
EXPECTED := ['Egli chinò la testa, senza togliere lo sguardo dalla bambina e dal cane, e disse:']
PREDICTED := ['Egli si alzò , non si , si dalla poltrona e la vecchia bambina .']
Validation cer: 0.6419752836227417
Validation wer: 0.800000011920929
Validation BLEU: 0.0
--------------------
Epoch  5
Train Loss: 4.480361
Validation Loss: 4.652407
------
SOURCE := [When the wife left the box the husband loitered behind, trying to catch Anna's eye and evidently wishing to bow to her.]
EXPECTED := ['Quando la moglie uscì, il marito si attardò a lungo, cercando con gli occhi lo sguardo di Anna, con l’evidente desiderio di salutarla.']
PREDICTED := ['Quando la moglie si avvicinò alla contessa , si mise a sedere , cercando di e si mise a .']
Validation cer: 0.611940324306488
Validation wer: 0.8260869383811951
Validation BLEU: 0.0
--------------------
Epoch  6
Train Loss: 4.246380
Validation Loss: 4.611298
------
SOURCE := [When the wife left the box the husband loitered behind, trying to catch Anna's eye and evidently wishing to bow to her.]
EXPECTED := ['Quando la moglie uscì, il marito si attardò a lungo, cercando con gli occhi lo sguardo di Anna, con l’evidente desiderio di salutarla.']
PREDICTED := ['Quando la moglie si avvicinò alla stazione , Aleksej Aleksandrovic si mise a , cercando di l ’ occhio e , evidentemente , si mise a .']
Validation cer: 0.611940324306488
Validation wer: 0.9130434989929199
Validation BLEU: 0.0
--------------------
Epoch  7
Train Loss: 4.015939
Validation Loss: 4.594482
------
SOURCE := [“However, my old friend,” says he, “you shall not want a supply in your necessity; and as soon as my son returns you shall be fully satisfied.” Upon this he pulls out an old pouch, and gives me one hundred and sixty Portugal moidores in gold; and giving the writings of his title to the ship, which his son was gone to the Brazils in, of which he was quarter-part owner, and his son another, he puts them both into my hands for security of the rest.]
EXPECTED := ['Ciò non ostante il mio buon capitano confessò d’andarmi debitore di quattrocento settanta moidori d’oro oltre al valore di sessanta casse di zucchero, e di quindici doppi rotoli di tabacco, le quali mercanzie avea perdute insieme con la nave che le portava, per un naufragio cui quel poveretto soggiacque nel tornare a Lisbona undici anni dopo la mia partenza. Qui mi raccontò come si trovasse costretto a valersi del mio danaro, per riparare i sofferti danni e comperarsi una parte di proprietà in altro vascello mercantile.']
PREDICTED := ['« Ma il mio vecchio capitano non si può , e non potete dir vero che la vostra vita non in me ; e la vostra storia si di questo figlio , e di e di in un ’ antica piantagione , e di in un ’ altra parte del figlio , e di che fu il capitano . — « La mia piantagione , in un ’ altra parte , il mio figlio , mi ha fatto che il mio figlio si , e la mia scialuppa , e la mia storia .']
Validation cer: 0.691428542137146
Validation wer: 1.0116279125213623
Validation BLEU: 0.0
--------------------
Epoch  8
Train Loss: 3.793747
Validation Loss: 4.627824
------
SOURCE := [Was it suspected that this lunatic, Mrs. Rochester, had any hand in it?"]
EXPECTED := ['i sospetti non son caduti sulla pazza?']
PREDICTED := ['È forse che quella signora Fairfax , che era stata in mano ?']
Validation cer: 1.1578947305679321
Validation wer: 1.8571428060531616
Validation BLEU: 0.0
--------------------
Epoch  9
Train Loss: 3.579768
Validation Loss: 4.650926
------
SOURCE := ['Because it means going goodness knows where, and by what roads! to what inns!]
EXPECTED := ['— Perché andare Dio sa dove, chi sa per quali strade, in quali alberghi.']
PREDICTED := ['— Perché lo , dove ne parla , e che cosa si tratta di questa primavera !']
Validation cer: 0.6666666865348816
Validation wer: 1.0714285373687744
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
 
