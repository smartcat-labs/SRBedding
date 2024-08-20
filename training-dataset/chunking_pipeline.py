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

def return_dic(sentences : List[str]) -> List[Dict[str, str|int]]:
    """
    Generates a list of dictionaries from a list of sentences.

    Each dictionary contains a sentence and its corresponding index in the original list.

    Parameters:
    sentences (List[str]): A list of sentences.

    Returns:
    List[Dict[str, str|int]]: A list where each element is a dictionary with keys:
        - 'sentence' [str]: The original sentence.
        - 'id' [int]: The index of the sentence in the input list.
         Example:
        >>> texts = ["Sentence 1" "Sentence 2"]
        >>> sentences_dic = return_dic(texts)
        >>> print(sentences_dic)
    """
    sent_lst_dic = [{'sentence': x, 'id' : i} for i, x in enumerate(sentences)]
    return sent_lst_dic

def combine_sentences(sentences: List[Dict[str,str]], buffer_size: int) -> List[Dict[str,str]]:
    """
    Combines each sentence in a list with its surrounding sentences based on a buffer size.

    For each sentence, the function concatenates the sentences within the specified buffer size before and after it, 
    creating a new key 'combined_sentence' in each dictionary.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'sentence' key.
    buffer_size (int): The number of sentences before and after the current sentence to include in the combination.

    Returns:
    List[Dict[str, str]]: The updated list of dictionaries with each dictionary containing an additional 'combined_sentence' key.
     Example:
    >>> texts = ["Sentence 1", "Sentence 2"]
    >>> sentences_dic = return_dic(texts)
    >>> print(sentences_dic)
    """
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
    """
    Generates embeddings for combined sentences and adds them to each sentence dictionary.

    The function uses the `oaiembeds.embed_documents` method to generate embeddings for the 'combined_sentence' 
    and appends the resulting embeddings as a new key 'combined_sentence_embedding' in each dictionary.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'combined_sentence' key.

    Returns:
    List[Dict[str, str]]: The updated list of dictionaries with an additional 'combined_sentence_embedding' key.

    Example:
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.', 'combined_sentence': 'This is the first sentence. This is the second sentence.'},
    ...     {'sentence': 'This is the second sentence.', 'combined_sentence': 'This is the first sentence. This is the second sentence. This is the third   sentence.'}
    ... ]
    >>> embeddings = generate_embeddings(sentences)
    >>> print(embeddings)
    """
    embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    
    return sentences

def calculate_cosine_distances(sentences: List[Dict[str,str]]) -> tuple[List[int], List[str]]:
    """
    Calculates cosine distances between the embeddings of consecutive sentences and adds the distance to each sentence dictionary.

    The function computes the cosine distance between the 'combined_sentence_embedding' of each sentence and the next one in the list.
    It appends the distance as a new key 'distance_to_next' in each dictionary and returns the list of distances along with the updated list of dictionaries.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'combined_sentence_embedding' key.

    Returns:
    Tuple[List[float], List[Dict[str, str]]]: 
        - A list of cosine distances between consecutive sentence embeddings.
        - The updated list of dictionaries with an additional 'distance_to_next' key.

    Example:
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.', 'combined_sentence_embedding': [0.1, 0.2, 0.3]},
    ...     {'sentence': 'This is the second sentence.', 'combined_sentence_embedding': [0.4, 0.5, 0.6]}]
    >>> distances, updated_sentences = calculate_cosine_distances(sentences)
    >>> print(distances)
    >>> print(updated_sentences)
    """
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
    """
    Splits a list of sentences into chunks based on a distance threshold. 
    If you want more chunks, lower the percentile cutoff.

    The function identifies breakpoints in the sentence list where the cosine distances between sentence embeddings 
    exceed a specified percentile threshold. It then splits the sentences into chunks at those breakpoints.

    Parameters:
    distances (List[int]): A list of cosine distances between consecutive sentence embeddings.
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'sentence' key.
    threshold (int): The percentile threshold for determining breakpoints. Lower thresholds result in more chunks.

    Returns:
    List[str]: A list of combined text chunks, where each chunk is formed by joining sentences that fall between breakpoints.

    Example:
    >>> distances = [0.1, 0.4, 0.7, 0.2]
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.'},
    ...     {'sentence': 'This is the second sentence.'},
    ...     {'sentence': 'This is the third sentence.'}
    ... ]
    >>> chunks = get_breakpoint(distances, sentences, 50)
    >>> print(chunks)
    """
    breakpoint_distance_threshold = np.percentile(distances, threshold)
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
    """
    Calculates the number of tokens in a string based on a specified encoding.

    The function uses the specified encoding to tokenize the input string and returns the number of tokens generated.

    Parameters:
    string (str): The input string to be tokenized.
    encoding_name (str): The name of the encoding to use for tokenization.

    Returns:
    int: The number of tokens in the input string based on the specified encoding.

    Example:
    >>> string = "This is a sample sentence."
    >>> encoding_name = "cl100k_base"
    >>> num_tokens = num_tokens_from_string(string, encoding_name)
    >>> print(num_tokens)
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_threshold(distances: List[int], sentences: List[str]) -> int:
    """
    Determines the optimal threshold value for splitting sentences into chunks based on token length.

    The function iterates over a range of threshold values, using each one to generate chunks of sentences.
    It then calculates the median token length of the chunks and returns the first threshold that results in 
    a median chunk length greater than 90 tokens. If no such threshold is found, it returns 90.

    Parameters:
    distances (List[float]): A list of cosine distances between consecutive sentence embeddings.
    sentences (List[str]): A list of sentence strings to be chunked.

    Returns:
    int: The optimal threshold value that results in chunks with a median token length greater than 90.

    Example:
    >>> distances = [0.1, 0.4, 0.7, 0.2]
    >>> sentences = [
    ...     "This is the first sentence.",
    ...     "This is the second sentence.",
    ...     "This is the third sentence."
    ... ]
    >>> threshold = get_threshold(distances, sentences)
    >>> print(threshold)
    """
    for threshold in range(40, 100, 5):
        breaks = get_breakpoint(distances=distances, sentences=sentences, threshold=threshold)
        
        chunk_lengths = []
        for chunk in breaks:
            tokens = num_tokens_from_string(chunk, 'cl100k_base')
            chunk_lengths.append(tokens)

        median_length = median(chunk_lengths)

        if median_length > 110:
            return threshold
 
    return 90

def split_chunk(big_chunk: str, smallest_size: int, largest_size: int) -> List[str]:
    """
    Splits a large chunk of text into smaller chunks based on token size.

    The function divides a large chunk of text into smaller chunks by splitting the text at sentence boundaries.
    It ensures that each resulting chunk has a token size between 50 and 450 tokens.
    If a chunk exceeds 450 tokens, the function discards that chunk and moves on.

    Parameters:
    big_chunk (str): The large chunk of text to be split.

    Returns:
    List[str]: A list of smaller text chunks that each contain between 50 and 450 tokens.

    Example:
    >>> big_chunk = "This is the first sentence. This is the second sentence. This is the third sentence."
    >>> small_chunks = split_chunk(big_chunk)
    >>> print(small_chunks)
    """
    splits = []
    sentences = big_chunk.split(".")
    current_sentence = ""
    for sentence in sentences:
        token_size = num_tokens_from_string(current_sentence, 'cl100k_base')
        if smallest_size < token_size <largest_size:
            splits.append(current_sentence)
            current_sentence = ''
        elif token_size>=largest_size:
            current_sentence = ""
        else:
            current_sentence += sentence
    return splits

def get_filtered_chunks(chunks: List[str]) -> List[str]:
    """
    Filters and refines chunks of text based on token size.

    The function processes a list of text chunks, filtering out those with fewer than 50 tokens and splitting 
    those with more than 450 tokens into smaller chunks. It returns a list of chunks that each contain between 
    50 and 450 tokens.

    Parameters:
    chunks (List[str]): A list of text chunks to be filtered and refined.

    Returns:
    List[str]: A list of text chunks, each containing between 50 and 450 tokens.

    Example:
    >>> chunks = [
    ...     "This is a short chunk.",
    ...     "This is a longer chunk that should be included because it has a moderate number of tokens.",
    ...     "This is a very long chunk that exceeds 450 tokens and needs to be split into smaller chunks."
    ... ]
    >>> filtered_chunks = get_filtered_chunks(chunks)
    >>> print(filtered_chunks)
    """
    smallest_size = 175
    largest_size = 500
    filtered = []
    for chunk in chunks:
        token_num = num_tokens_from_string(chunk, 'cl100k_base')
        if token_num < smallest_size:
            continue
        if token_num < largest_size:
            filtered.append(chunk)
        else:
            filtered.extend(split_chunk(chunk, smallest_size, largest_size))
    return filtered 

def get_chunks(sentences: List[str], buffer_size: int) -> List[str]:
    """
    Generates optimized text chunks from a list of sentences using cosine distances and token thresholds.

    This function processes a list of sentences by performing several steps:
    1. Converts sentences into dictionaries with IDs.
    2. Combines sentences based on a specified buffer size.
    3. Generates embeddings for the combined sentences.
    4. Calculates cosine distances between the embeddings of consecutive sentences.
    5. Determines an optimal threshold for splitting sentences into chunks.
    6. Breaks the sentences into chunks based on the threshold.
    7. Filters and refines the chunks based on token size.

    Parameters:
    sentences (List[str]): A list of sentences to be processed into chunks.
    buffer_size (int): The number of neighboring sentences to consider when combining sentences.

    Returns:
    List[str]: A list of optimized text chunks.

    Example:
    >>> sentences = [
    ...     "This is the first sentence.",
    ...     "This is the second sentence.",
    ...     "This is the third sentence.",
    ...     "This is the fourth sentence.",
    ...     "This is the fifth sentence."
    ... ]
    >>> buffer_size = 2
    >>> chunks = get_chunks(sentences, buffer_size)
    >>> print(chunks)
    """
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
    

    contexts = ["Pajton je veoma popularan programski jezik opšte namene. Postao je poznat po svojoj jednostavnosti, lakoći učenja i brzini programiranja.     Mnogi profesionalni programeri koriste Pajton bar kao pomoćni jezik, jer pomoću njega brzo i lako automatizuju razne poslove. ",
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