import os
from typing import Dict, List
import numpy as np
import numpy as np
import openai
import random
import tiktoken

from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from statistics import median
from langchain_community.embeddings import OpenAIEmbeddings

oaiembeds = OpenAIEmbeddings(model='text-embedding-3-small')

def return_dic(sentences):
    sent_lst_dic = [{'sentence': x, 'id' : i} for i, x in enumerate(sentences)]
    return sent_lst_dic

def combine_sentences(sentences: List[Dict[str,str]], buffer_size: int) -> List[Dict[str,str]]:
    for i in range(len(sentences)):
        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '
        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def generate_embeddings(sentences: List[Dict[str,str]]) -> List[Dict[str,str]]:
    embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    
    return sentences

def calculate_cosine_distances(sentences: List[Dict[str,str]]) -> tuple[List[int], List[str]]:
    distances = []
    
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)

        sentences[i]['distance_to_next'] = distance

    return distances, sentences

def get_breakpoint(distances: List[int], sentences: List[str], threshold: int) -> List[str]:

    breakpoint_distance_threshold = np.percentile(distances, threshold) # If you want more chunks, lower the percentile cutoff
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    start_index = 0
    chunks = []

    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        
        start_index = index + 1

    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_threshold(distances: List[int], sentences: List[str]) -> int:
    for threshold in range(40, 100, 5):
        breaks = get_breakpoint(distances=distances, sentences=sentences, threshold=threshold)
        
        chunk_lengths = []
        for chunk in breaks:
            tokens = num_tokens_from_string(chunk, 'cl100k_base')
            chunk_lengths.append(tokens)

        median_length = median(chunk_lengths)

        if median_length > 90:
            return threshold
 
    return 90

def split_chunk(big_chunk: str) -> List[str]:
    splits = []
    sentences = big_chunk.split(".")
    current_sentence = ""
    for sentence in sentences:
        token_size = num_tokens_from_string(current_sentence, 'cl100k_base')
        if 50 < token_size <450:
            splits.append(current_sentence)
            current_sentence = ''
        elif token_size>=450:
            current_sentence = ""
        else:
            current_sentence += sentence
    return splits

def get_filtered_chunks(chunks: List[str]) -> List[str]:
    filtered = []
    for chunk in chunks:
        token_num = num_tokens_from_string(chunk, 'cl100k_base')
        if token_num < 50:
            continue
        if token_num < 450:
            filtered.append(chunk)
        else:
            filtered.extend(split_chunk(chunk))
    return filtered 

def get_chunks(sentences: List[str], buffer_size: int) -> List[str]:
    sentences_dic = return_dic(sentences)
    sentences_comb = combine_sentences(sentences_dic, buffer_size)
    sentences_embed = generate_embeddings(sentences_comb)
    distances, sentences = calculate_cosine_distances(sentences_embed)
    threshold = get_threshold(distances, sentences)
    sentences_breaks = get_breakpoint(distances, sentences, threshold)
    chunks = get_filtered_chunks(sentences_breaks)
    return chunks

if __name__== "__main__":

    # set environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    

    contexts = ["Pajton je veoma popularan programski jezik opšte namene. Postao je poznat po svojoj jednostavnosti, lakoći učenja i brzini programiranja. Mnogi profesionalni programeri koriste Pajton bar kao pomoćni jezik, jer pomoću njega brzo i lako automatizuju razne poslove. ",
                 "Za izvršavanje programa koje pišemo na Pajtonu, potreban nam je program koji se zove Pajton interpreter. Ovaj program tumači (interpretira), a zatim i izvršava Pajton naredbe. Pajton interpreteri mogu da prihvate cele programe i da ih izvrše, a mogu da rade i u interaktivnom režimu, ",
                 "Još jedan način da pokrenete Pajton školjku je da otvorite komandni prozor (na Windows sistemima to se radi pokretanjem programa cmd), a zatim u komandnom prozoru otkucate Python (ovde podrazumevamo da je Pajton instaliran tako da je dostupan iz svakog foldera, u protivnom treba se prvo pozicionirati u folder u kome se nalazi Pajton interpreter).",
                 "Novi Sad je Evropska prestonica kulture 2022. Novi Sad je, posle Beograda, drugi grad u Srbiji po broju stanovnika (bez podataka za područje Kosova i Metohije). Na poslednjem zvaničnom popisu iz 2011. godine, sam grad je imao 231.798[2] stanovnika. Na opštinskom području Novog Sada (uključujući i prigradska nasenja) broj stanovnika je 2011. godine iznosio 341.625.[",
                 "Najstariji arheološki ostaci (iz vremena kamenog doba) pronađeni su sa obe strane Dunava, na području današnjeg Petrovaradina (koji je u kontinuitetu nastanjen od praistorije do danas) i području današnje Klise. Istraživanjem ostataka naselja iz mlađeg bronzanog doba (3000. godina pne.) na području današnjeg Petrovaradina, arheolozi su pronašli i bedeme pojačane koljem i palisadama iz tog perioda, koji svedoče da je još u vreme vučedolske kulture ovde postojalo utvrđeno naselje.",
                 "Sredinom 6. veka, područje naseljavaju Sloveni. U 9. veku, tvrđava Petrikon (ili na slovenskim jezicima - Petrik) ulazi u sastav Bugarskog carstva, a u 11. veku njom vlada sremski vojvoda Sermon, čiji su zlatnici u 19. veku pronađeni u jednom petrovaradinskom vinogradu. Pošto je Bugarska poražena od Vizantije a Sermon ubijen, tvrđava ponovo postaje deo Vizantije, da bi, posle borbi Vizantinaca i Mađara, krajem 12. veka, ušla u sastav srednjovekovne Kraljevine Ugarske. Na području Bačke, ugarska vlast se ustaljuje nešto ranije, tokom 10. veka.",
                 "U vreme Osmanske uprave poznat pod imenom Varadin, Petrovaradin je bio sedište nahije u okviru Sremskog sandžaka. Podgrađe tvrđave imalo je oko 200 kuća, tu se nalazila Sulejman-hanova džamija, a postojale su i dve manje džamije, Hadži-Ibrahimova i Huseinova. Pored dve turske mahale, u sastavu grada nalazila se i hrišćanska četvrt sa 35 kuća, naseljenih isključivo Srbima.[8] Od praistorije pa sve do kraja 17. veka, centar urbanog života na području današnjeg grada nalazio se na sremskoj strani Dunava, na prostoru današnjeg Petrovaradina, koji je svojim značajem uvek zasenjivao naselja na bačkoj strani.",
                 "Tokom većeg dela 18. i 19. veka Novi Sad je bio središte kulturnog, političkog i društvenog života celokupnog srpskog naroda, koji tada nije imao sopstvenu državu, a prema podacima iz prve polovine 19. veka, ovo je bio i najveći grad nastanjen Srbima (Oko 1820. godine Novi Sad je imao oko 20.000 stanovnika, od kojih su oko dve trećine bili Srbi, a današnji najveći srpski grad, Beograd, nije dostigao približan broj stanovnika pre 1853. godine). Zbog svog kulturnog i političkog uticaja, Novi Sad je postao poznat kao Srpska Atina.",
                 "Posle sprovedenih izbora po svim vojvođanskim mestima (od 18. do 24. novembra), u Novom Sadu se 25. novembra 1918. godine sastala Velika narodna skupština Srba, Bunjevaca i ostalih Slovena Banata, Bačke i Baranje, koja zvanično proglašava otcepljenje ovih regiona od Ugarske i njihovo prisajedinjenje Srbiji, a na istoj skupštini formira se i pokrajinska vlada (Narodna uprava) Banata, Bačke i Baranje sa sedištem u Novom Sadu. 1. decembra 1918. godine, proglašeno je Kraljevstvo Srba, Hrvata i Slovenaca, a Novi Sad ulazi u tu novu državu kao sastavni deo Kraljevine Srbije.",
                 " Novom Sadu je, tokom celog rata, delovao pokret otpora i oslobodilački pokret pod vođstvom Komunističke Partije Jugoslavije. U gradu se nalazilo sedište okružnog komiteta Komunističke Partije Jugoslavije, koji je u okviru narodnooslobodilačkog pokreta bio deo Pokrajinskog komiteta KPJ za Vojvodinu. U narodnooslobodilačkoj borbi, u partizanskim odredima i vojvođanskim brigadama, neposredno je učestvovalo 2.365 Novosađana, pripadnika svih nacionalnosti, (Srba, Mađara, Slovaka i ostalih)"
                 ]
    sentences = []
    for context in contexts:
        sentences.extend(context.split("."))
    chunks = get_chunks(sentences=sentences, buffer_size=1)
    print(len(chunks))
    pprint(chunks)