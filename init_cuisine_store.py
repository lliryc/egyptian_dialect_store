from langchain_community.vectorstores import LanceDB
import lancedb
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI
import wikipedia

def get_wikipedia_page_content(title, language_code='arz'):
    try:
        wikipedia.set_lang(language_code)
        page = wikipedia.page(title)
        return page.content
    except wikipedia.exceptions.PageError:
        print(f"Page not found: {title}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation Error: {e.options}")
        return None

documents = []
documents_en = []

# Egyptian cuisine
doc_food = { "text": get_wikipedia_page_content("مطبخ مصرى"), "category": "egyptian cuisine", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D9%85%D8%B7%D8%A8%D8%AE_%D9%85%D8%B5%D8%B1%D9%89" }
documents.append(doc_food)

doc_food_en = { "text": get_wikipedia_page_content("Egyptian cuisine", language_code="en"), "category": "egyptian cuisine", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Egyptian_cuisine" }
documents_en.append(doc_food_en)


doc_cheese = { "text": get_wikipedia_page_content("الجبنه المصريه"), "category": "egyptian cuisine", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D8%A7%D9%84%D8%AC%D8%A8%D9%86%D8%A9_%D8%A7%D9%84%D9%85%D8%B5%D8%B1%D9%8A%D8%A9" }
documents.append(doc_cheese)

doc_cheese_en = { "text": get_wikipedia_page_content("Egyptian cheese", language_code="en"), "category": "egyptian cuisine", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Egyptian_cheese" }
documents_en.append(doc_cheese_en)

# Famous people 
doc_famous_people = { "text": get_wikipedia_page_content("شخصيات مصريه مشهوره"), "category": " egyptian famous people", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D8%B4%D8%AE%D8%B5%D9%8A%D8%A7%D8%AA_%D9%85%D8%B5%D8%B1%D9%8A%D9%87_%D9%85%D8%B4%D9%87%D9%88%D8%B1%D9%87" }
documents.append(doc_famous_people)

doc_famous_people_en = { "text": get_wikipedia_page_content("List of Egyptians", language_code="en"), "category": " egyptian famous people", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/List_of_Egyptians" }
documents_en.append(doc_famous_people_en)

# Cinema
doc_cinema2017 = { "text": get_wikipedia_page_content("الافلام المصريه لسنة 2017"), "category": "egyptian cinema", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D8%A7%D9%84%D8%A7%D9%81%D9%84%D8%A7%D9%85_%D8%A7%D9%84%D9%85%D8%B5%D8%B1%D9%8A%D9%87_%D9%84%D8%B3%D9%86%D8%A9_2017" }
documents.append(doc_cinema2017)

doc_cinema2017_en = { "text": get_wikipedia_page_content("List of Egyptian films of 2017", language_code="en"), "category": "egyptian cinema", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/List_of_Egyptian_films_of_2017" }
documents_en.append(doc_cinema2017_en)

doc_cinema =  { "text": get_wikipedia_page_content("سينما مصريه"), "category": "egyptian cinema", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D8%B3%D9%8A%D9%86%D9%85%D8%A7_%D9%85%D8%B5%D8%B1%D9%8A%D9%87" }
documents.append(doc_cinema)

doc_cinema_en = { "text": get_wikipedia_page_content("Cinema of Egypt", language_code="en"), "category": "egyptian cinema", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Cinema_of_Egypt" }
documents_en.append(doc_cinema_en)

# World Heritage Sites
doc_world_heritage = { "text": get_wikipedia_page_content("مواقع التراث العالمى فى مصر"), "category": "egyptian world heritage sites", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D9%85%D9%88%D8%A7%D9%82%D8%B9_%D8%A7%D9%84%D8%AA%D8%B1%D8%A7%D8%AB_%D8%A7%D9%84%D8%B9%D8%A7%D9%84%D9%85%D9%89_%D9%81%D9%89_%D9%85%D8%B5%D8%B1" }
documents.append(doc_world_heritage)

doc_world_heritage_en = { "text": get_wikipedia_page_content("List of World Heritage Sites in Egypt", language_code="en"), "category": "egyptian world heritage sites", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_Egypt" }
documents_en.append(doc_world_heritage_en)


# Museums
doc_museums = { "text": get_wikipedia_page_content("ليستة المتاحف المصريه"), "category": "egyptian museums", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D9%84%D9%8A%D8%B3%D8%AA%D8%A9_%D8%A7%D9%84%D9%85%D8%AA%D8%A7%D8%AD%D9%81_%D8%A7%D9%84%D9%85%D8%B5%D8%B1%D9%8A%D9%87" }
documents.append(doc_museums)

doc_museums_en = { "text": get_wikipedia_page_content("List of museums in Egypt", language_code="en"), "category": "egyptian museums", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/List_of_museums_in_Egypt" }
documents_en.append(doc_museums_en)

# Music
doc_music = { "text": get_wikipedia_page_content("موسيقى مصريه"), "category": "egyptian music traditions", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D9%85%D9%88%D8%B3%D9%8A%D9%82%D9%89_%D9%85%D8%B5%D8%B1%D9%8A%D9%87" }
documents.append(doc_music)

doc_music_en = { "text": get_wikipedia_page_content("Music of Egypt", language_code="en"), "category": "egyptian music traditions", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Music_of_Egypt" }
documents_en.append(doc_music_en)

# Egyptian Culture
doc_culture = { "text": get_wikipedia_page_content("الثقافه فى مصر"), "category": "egyptian culture", "source": "wikipedia", "url": "https://arz.wikipedia.org/wiki/%D8%A7%D9%84%D8%AB%D9%82%D8%A7%D9%81%D8%A9_%D9%81%D9%89_%D9%85%D8%B5%D8%B1" }
documents.append(doc_culture)

doc_culture_en = { "text": get_wikipedia_page_content("Culture of Egypt", language_code="en"), "category": "egyptian culture", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Culture_of_Egypt" }
documents_en.append(doc_culture_en)

def validate_documents(docs):
    valid_docs = []
    for doc in docs:
        if doc["text"] is not None:
            # Convert dictionary to Document object
            valid_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "category": doc["category"],
                        "source": doc["source"],
                        "url": doc["url"]
                    }
                )
            )
    return valid_docs

validated_documents = validate_documents(documents)
validated_documents_en = validate_documents(documents_en)

if __name__ == "__main__":

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # init database connection
    db = lancedb.connect('lancedb')

    # Create table through the database connection instead of directly using the table
    docsearch = LanceDB.from_documents(validated_documents, embedding, connection=db, table_name="egyptian_culture_nuances")
    docsearch_en = LanceDB.from_documents(validated_documents_en, embedding, connection=db, table_name="egyptian_culture_nuances_en")
