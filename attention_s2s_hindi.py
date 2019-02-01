from keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Bidirectional\
,Lambda,Dot,RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras.backend as K

import numpy as np
import matplotlib.pyplot  as plt

latent_dim=256
embedding_dim=100


english=[]
hindi_input=[]
hindi_output=[]
i=0

with open("hin-eng/hin.txt",encoding='utf8') as f:
    for line in f:
#        if i>1500:
#            break
#        i+=1
        eng,hin=line.rstrip().split('\t')
        english.append(eng)
        a='<sos> '+hin
        b=hin+' <eos>'
        hindi_input.append(a)
        hindi_output.append(b)
with open('C:/Users/Naveen Kumar/Downloads/dev_test/dev_test/hin.txt',encoding='utf8') as f:
    a=[]
    for line  in f:
        p='<sos> '+line
        q=line+' <eos>'
        hindi_input.append(p)
        hindi_output.append(q)
with open('C:/Users/Naveen Kumar/Downloads/dev_test/dev_test/dev.txt',encoding='utf8') as f:
    a=[]
    for line  in f:
        english.append(line)
        


tokenizer=Tokenizer(num_words=20000)
tokenizer.fit_on_texts(english)
encoder_data=tokenizer.texts_to_sequences(english)
encoder_word2idx=tokenizer.word_index
encoder_length=min(20,max(len(s) for s in encoder_data))
encoder_vocab_size=len(encoder_word2idx)+1


tokenizer=Tokenizer(num_words=20000,filters='')
tokenizer.fit_on_texts(hindi_input+hindi_output)
decoder_input_data=tokenizer.texts_to_sequences(hindi_input)
decoder_output_data=tokenizer.texts_to_sequences(hindi_output)
decoder_length=min(20,max(len(s) for s in decoder_input_data))
decoder_word2idx=tokenizer.word_index
decoder_vocab_size=len(decoder_word2idx)+1

print(len(decoder_input_data),decoder_vocab_size,decoder_length)

encoder_data=pad_sequences(encoder_data,maxlen=encoder_length)
decoder_input_data=pad_sequences(decoder_input_data,maxlen=decoder_length,padding='post')
decoder_output_data=pad_sequences(decoder_output_data,maxlen=decoder_length,padding='post')

decoder_targets=np.zeros((len(decoder_input_data),decoder_length,decoder_vocab_size),dtype="float32")

for i,output in enumerate(decoder_output_data):
    for j,out in enumerate(output):
        decoder_targets[i,j,out]=1

embedding_matrix=np.zeros((encoder_vocab_size,embedding_dim),dtype="float32")
word2vec = {}
with open('glove.6B.100d.txt',encoding='utf8') as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))




# prepare embedding matrix
print('Filling pre-trained embeddings...')
for word, i in encoder_word2idx.items():
  if i < 20000:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector
      
embedding_layer=Embedding(encoder_vocab_size,embedding_dim,weights=[embedding_matrix],input_length=encoder_length)

encoder_placeholder=Input(shape=(encoder_length,))
x=embedding_layer(encoder_placeholder)
encoder_lstm=Bidirectional(LSTM(latent_dim,return_sequences=True))
encoder_outputs=encoder_lstm(x)

def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

decoder_placeholder=Input(shape=(decoder_length,))
decoder_embedding=Embedding(decoder_vocab_size,latent_dim)
decoder_inputs_x=decoder_embedding(decoder_placeholder)
decoder_lstm=LSTM(latent_dim,return_state=True)
dense=Dense(decoder_vocab_size,activation="softmax")
context_last_word_concat=Concatenate(axis=2)

initial_s=Input(shape=(latent_dim,))
initial_c=Input(shape=(latent_dim,))

s=initial_s
c=initial_c

attn_repeat=RepeatVector(encoder_length)
attn_concat=Concatenate(axis=-1)
attn_dense1=Dense(10,activation='tanh')
attn_dense2=Dense(1,activation=softmax_over_time)
dot=Dot(axes=1)

def one_step_attention(e,s):
    a=attn_repeat(s)
    b=attn_concat([e,a])
    c=attn_dense1(b)
    d=attn_dense2(c)
    e=dot([d,e])
    return e
outputs=[]
for i in range(decoder_length):
    context=one_step_attention(encoder_outputs,s)
    selector=Lambda(lambda x:x[:,i:i+1])
    xt=selector(decoder_inputs_x)
    decoder_lstm_input=context_last_word_concat([context,xt])
    decoder_outputs,s,c=decoder_lstm(decoder_lstm_input,initial_state=[s,c])
    output=dense(decoder_outputs)
    outputs.append(output)
def stacker(x):
    x=K.stack(x)
    x=K.permute_dimensions(x,pattern=(1,0,2))
    return x
stacker1=Lambda(stacker)
output_=stacker1(outputs)

model=Model([encoder_placeholder,decoder_placeholder,initial_s,initial_c],output_)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
r=model.fit([encoder_data,decoder_input_data,np.zeros((len(decoder_input_data),latent_dim),dtype="float32"),np.zeros((len(decoder_input_data),latent_dim),dtype="float32")],decoder_targets,validation_split=0.2,batch_size=64,epochs=5)
    
sampling_encoder_model=Model(encoder_placeholder,encoder_outputs)

sampling_decoder_input=Input(shape=(1,))
sampling_decoder_inputs_x=decoder_embedding(sampling_decoder_input)
sampling_encoder_inputs=Input(shape=(encoder_length,latent_dim*2))
context=one_step_attention(sampling_encoder_inputs,initial_s)
sampling_decoder_lstm_input=context_last_word_concat([context,sampling_decoder_inputs_x])
sampling_decoder_outputs,s,c=decoder_lstm(sampling_decoder_lstm_input,initial_state=[initial_s,initial_c])
sampling_output=dense(sampling_decoder_outputs)

sampling_decoder_model=Model([sampling_decoder_input,sampling_encoder_inputs,initial_s,initial_c],[sampling_output,s,c])

e={v:k for k,v in decoder_word2idx.items()}
def decode_sequence(input_seq):
  # Encode the input as state vectors.
  enc_out = sampling_encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))
  
  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = decoder_word2idx['<sos>']

  # if we get this we break
  eos = decoder_word2idx['<eos>']


  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, latent_dim))
  c = np.zeros((1, latent_dim))


  # Create the translation
  output_sentence = []
  for _ in range(decoder_length):
    o, s, c = sampling_decoder_model.predict([target_seq, enc_out, s, c])
        

    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = e[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)




  x=input()
  t=Tokenizer(num_words=20000)
  t.fit_on_texts(english)
  s=t.texts_to_sequences([x])
  s=pad_sequences(s,maxlen=encoder_length)
  translation = decode_sequence(s)
  
  
        



