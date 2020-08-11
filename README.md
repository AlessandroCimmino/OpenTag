# OpenTag

##File da inserire per l'esecuzione
Inserire all'interno del file config il path al dataset del benchmark sostituendolo
al valore di "DIRECTORY_DATASET".

Scaricare gli embeddings di GloVe, inserirli all'interno della cartella persistent_files
e modificare coerentemente la voce "GLOVE_WORD_EMBD" del file config.

Scaricare il modello pre addestrato di BERT e modificare la voce "PRETRAINED_MODEL"
del file config.

##Esecuzione
Per decide se utilizzare il modello con GloVe o BERT basta eseguire opentag_bert.py o opentag_glove.py

Prima di eseguire bisogna modificare alcuni voci del file config:

-Modificare la voce "ATTRIBUTES" del config con i target attribute su cui si vogliono
effettuare gli esperimenti. Modificare anche la voce "TAGS" nel caso in cui mancassero i
tag relativi a quell'attributo.

-Definire il file per effettuare il test modificando la voce "TEST_SET" del file config.

-Modificare la voce "RAF" per indicare se si vuole utilizzare o meno l'output di RAF
per la creazione del training set.
In caso contrario è possibile definire se si vuole creare da capo il dizionario dei valori
modificando "CREATE_DICTIONARY", se costruire training,valid e test set da capo modificando
"CREATE_TRAINING",se creare i train,valied e set come disjoint set modificando "DISJOINT" e se utilizzare tutte le sorgenti per la creazione dei dati di training modificando "ALL_SOURCES" (In caso negativo modificare "SOURCES" indicando le sorgenti da utilizzare).

-Modifcare "NEW_BERT_EMBD" se si vuole ottenere gli embeddings di BERT da capo.
(Nel caso in cui i dati di training siano molti potrebbe capitare che non si possibile salvare
  gli embeddings di BERT in locale, in questo caso si possono commentare le righe 134 e 133
  di opentag_bert.py)

-Modificare "NEW_TRAINING" per indicare se addestrare nuovamente il modello.
