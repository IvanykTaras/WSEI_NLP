from collections import Counter
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from nltk.corpus import stopwords 
from telegram import InputMediaPhoto, Update
from sentences import *
import shlex
import nltk
import re
import json

nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

TOKEN = "8622639294:AAGSsDW82owPsvm5vZVJ3zIk7_NnKbnzhLI"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your new bot.")

async def task(update: Update, context: ContextTypes.DEFAULT_TYPE):    
    if not update.message.text or update.message.text.strip() == "":
        await update.message.reply_text("❌ Error: Message is empty.")
        return
    
    try:
        Sentences.init_sentences()
    except Exception as e:
        await update.message.reply_text(f"❌ Error: Failed to load data from JSON. {str(e)}")
        return
    
    try:
        args = shlex.split(update.message.text)
        
        if len(args) < 4:
            await update.message.reply_text(
                "❌ Error: Invalid number of arguments.\n"
                "Usage: /task <command> <text> <class>\n\n"
                "Available commands:\n"
                "- tokenize\n"
                "- remove_stopwords\n"
                "- lemmatize\n"
                "- stemming\n"
                "- stats\n"
                "- n-grams\n"
                "- plot_histogram\n"
                "- plot_wordcloud"
            )
            return
        
        task_command = args[1]
        text = args[2]
        class_text = args[3]
        
        if not text or text.strip() == "":
            await update.message.reply_text("❌ Error: Text to process cannot be empty.")
            return
        
        valid_commands = [
            "tokenize", "remove_stopwords", "lemmatize", "stemming",
            "stats", "n-grams", "plot_histogram", "plot_wordcloud"
        ]
        
        if task_command not in valid_commands:
            await update.message.reply_text(
                f"❌ Error: Unknown command '{task_command}'.\n\n"
                f"Available commands:\n"
                f"- " + "\n- ".join(valid_commands)
            )
            return
    
    except IndexError:
        await update.message.reply_text(
            "❌ Error: Invalid command syntax.\n"
            "Usage: /task <command> <text> <class>"
        )
        return
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: Invalid arguments. {str(e)}")
        return

    if(task_command == "tokenize"):
        await update.message.reply_text(f"{nltk.word_tokenize(text)}")

    if(task_command == "remove_stopwords"):
        try:
            tokens = nltk.word_tokenize(text)
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            await update.message.reply_text(f"{filtered_tokens}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error during processing: {str(e)}")

    if(task_command == "lemmatize"):
        try:
            lemmatizer = nltk.WordNetLemmatizer()
            tokens = nltk.word_tokenize(text)
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            await update.message.reply_text(f"{lemmatized_tokens}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error during lemmatization: {str(e)}")

    if(task_command == "stemming"):
        try:
            stemmer = nltk.PorterStemmer()
            tokens = nltk.word_tokenize(text)
            stemmed_tokens = [stemmer.stem(word) for word in tokens]
            await update.message.reply_text(f"{stemmed_tokens}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error during stemming: {str(e)}")
    
    if task_command == "stats":
        try:
            tokens = nltk.word_tokenize(text) 
            char_count = len(text)
            word_count = len(tokens)
            avg_word_len = sum(len(w) for w in tokens) / word_count if word_count > 0 else 0
            
            result = (
                f"Sentence Statistics:\n"
                f"- Character count: {char_count}\n"
                f"- Word count: {word_count}\n"
                f"- Average word length: {avg_word_len:.2f}"
            )
            await update.message.reply_text(result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error during statistics calculation: {str(e)}")

    if task_command == "n-grams":
        try:
            tokens = nltk.word_tokenize(text)
            
            if len(tokens) < 2:
                await update.message.reply_text("❌ Error: Text must contain at least 2 words for n-grams.")
                return
            
            bigrams = list(nltk.ngrams(tokens, 2))
            trigrams = list(nltk.ngrams(tokens, 3))
            
            result = (
                f"Bigrams: {bigrams}\n\n"
                f"Trigrams: {trigrams}"
            )
            await update.message.reply_text(result)
        except Exception as e:
            await update.message.reply_text(f"❌ Error during n-gram generation: {str(e)}")

    if task_command == "plot_histogram":
        try:
            tokens = nltk.word_tokenize(text)
            lengths = [len(w) for w in tokens]
            
            if not lengths:
                await update.message.reply_text("❌ Error: Cannot generate histogram for empty text.")
                return
            
            plt.figure(figsize=(8, 6))
            plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), align='left', color='skyblue', edgecolor='black')
            plt.title(f"Token Length Histogram")
            plt.xlabel("Word Length")
            plt.ylabel("Occurrences")
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Sentence_{timestamp}.png"
            
            plt.savefig(filename)
            plt.close()
            
            with open(filename, 'rb') as photo:
                await update.message.reply_photo(photo)
        except Exception as e:
            await update.message.reply_text(f"❌ Error during histogram generation: {str(e)}")

    if task_command == "plot_wordcloud":
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off") 
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Sentence_{timestamp}.png"
            
            plt.savefig(filename)
            plt.close()
            
            with open(filename, 'rb') as photo:
                await update.message.reply_photo(photo)
        except Exception as e:
            await update.message.reply_text(f"❌ Error during wordcloud generation: {str(e)}")
            return

    try:
        Sentences.add_record(Record(text, class_text))
    except Exception as e:
        await update.message.reply_text(f"❌ Error: Failed to save record. {str(e)}")

async def full_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.text or update.message.text.strip() == "":
        await update.message.reply_text("❌ Error: Message is empty.")
        return
    
    try:
        args = shlex.split(update.message.text)
        
        if len(args) < 3:
            await update.message.reply_text(
                "❌ Error: Invalid number of arguments.\n"
                "Usage: /full_pipeline <text> <class>"
            )
            return
        
        text = args[1]
        class_text = args[2]
        
        if not text or text.strip() == "":
            await update.message.reply_text("❌ Error: Text cannot be empty.")
            return
    
    except (IndexError, ValueError):
        await update.message.reply_text(
            "❌ Error: Invalid command syntax.\n"
            "Usage: /full_pipeline <text> <class>"
        )
        return

    try:
        # 1. Clean text
        clean_text = re.sub(r'[^\w\s]', '', text).lower()

        # 2. Tokenize
        tokens = nltk.word_tokenize(clean_text)

        # 3. Remove stop words
        no_stop = [word for word in tokens if word not in stop_words]

        # 4. Lemmatize
        lemmatizer = nltk.WordNetLemmatizer()
        lems = [lemmatizer.lemmatize(w) for w in tokens]

        # 5. Stemming
        stemmer = nltk.PorterStemmer()
        stems = [stemmer.stem(w) for w in tokens]

        # 6. Bag of Words
        vectorizer = CountVectorizer()
        bow_rep = vectorizer.fit_transform([clean_text]).toarray().tolist()
        
        # 7. TF-IDF
        tfidf_vec = TfidfVectorizer()
        tfidf_rep = tfidf_vec.fit_transform([clean_text]).toarray().tolist()

        # 8. Statistics
        stat_char_count = len(text)
        stat_word_count = len(tokens)
        stat_avg_word_len = sum(len(w) for w in tokens) / stat_word_count if stat_word_count > 0 else 0
        
        stat_result = (
            f"Sentence Statistics:\n"
            f"- Character count: {stat_char_count}\n"
            f"- Word count: {stat_word_count}\n"
            f"- Average word length: {stat_avg_word_len:.2f}"
        )

        # 9. Generate charts
        lengths = [len(w) for w in tokens]
        
        if not lengths:
            await update.message.reply_text("❌ Error: Cannot process empty text.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), align='left', color='skyblue', edgecolor='black')
        plt.title(f"Token Length Histogram")
        plt.xlabel("Word Length")
        plt.ylabel("Occurrences")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_historgram_filename = f"histogram_Sentence_{timestamp}.png"
        
        plt.savefig(plot_historgram_filename)
        plt.close()
        

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off") 
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_wordcloud_filename = f"wordcloud_Sentence_{timestamp}.png"
        
        plt.savefig(plot_wordcloud_filename)
        plt.close()
        
        
        # 10. Present results
        report = (
            f"📝 **FULL PIPELINE REPORT**\n"
            f"━━━━━━━━━━━━━━━\n"
            f"1. **Text Cleaning**: `{clean_text}`\n"
            f"2. **Tokenization**: `{tokens}`\n"
            f"3. **Stop Words Removal**: `{no_stop}`\n"
            f"4. **Lemmatization**: `{lems}`\n"
            f"5. **Stemming**: `{stems}`\n"
            f"6. **Bag of Words**: `{bow_rep}`\n"
            f"7. **TF-IDF**: `{tfidf_rep}`\n"
            f"8. **Statistics**: \n\n```{stat_result}```\n\n"
            f"9. **Token Length Histogram**: `{plot_historgram_filename}`\n"
            f"10. **WordCloud**: `{plot_wordcloud_filename}`"
        )
        
        await update.message.reply_photo(open(plot_wordcloud_filename, 'rb'))
        await update.message.reply_photo(open(plot_historgram_filename, 'rb'))
        await update.message.reply_markdown(report)

        # 11. Add record to database
        Sentences.add_record(Record(text, class_text))
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error during full_pipeline processing: {str(e)}")
        return

async def classifier(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.text or update.message.text.strip() == "":
        await update.message.reply_text("❌ Error: Message is empty.")
        return
    
    try:
        args = shlex.split(update.message.text)
        
        if len(args) < 2:
            await update.message.reply_text(
                "❌ Error: Invalid number of arguments.\n"
                "Usage: /classifier <text>"
            )
            return
        
        text = args[1]
        
        if not text or text.strip() == "":
            await update.message.reply_text("❌ Error: Text cannot be empty.")
            return
        
        Sentences.init_sentences()
        
        if not Sentences.sentences or len(Sentences.sentences) == 0:
            await update.message.reply_text("❌ Error: Database is empty. Add some records first.")
            return
        
        texts = [record["text"] for record in Sentences.sentences]

        labels = []  

        for record in Sentences.sentences: 
            if record["classText"] == "positive":
                labels.append(1)        
            if record["classText"] == "neutral":
                labels.append(0)
            if record["classText"] == "negative":
                labels.append(-1)

        if not labels:
            await update.message.reply_text("❌ Error: Cannot build model without labels.")
            return

        model = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", LogisticRegression())
        ])

        model.fit(texts, labels)

        classText = ["negative", "neutral", "positive"]
        prediction = model.predict([text])[0]
        await update.message.reply_text(f"✅ Predicted class: {classText[prediction + 1]}")
        
    except IndexError:
        await update.message.reply_text(
            "❌ Error: Invalid command syntax.\n"
            "Usage: /classifier <text>"
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        await update.message.reply_text(f"❌ Error: Data problems. {str(e)}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error during classification: {str(e)}")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        Sentences.init_sentences()
        
        if not Sentences.sentences or len(Sentences.sentences) == 0:
            await update.message.reply_text("❌ Error: Database is empty. Add some records first.")
            return
        
        all_texts = [record.get("text", "") for record in Sentences.sentences]
        all_classes = [record.get("classText", "unknown") for record in Sentences.sentences]
        
        if not all_texts:
            await update.message.reply_text("❌ Error: No texts in database.")
            return
        
        full_blob = " ".join(all_texts).lower()
        all_tokens = nltk.word_tokenize(full_blob)
        unique_tokens = sorted(list(set([t for t in all_tokens if t.isalnum()])))

        bigrams = list(nltk.ngrams(all_tokens, 2))
        trigrams = list(nltk.ngrams(all_tokens, 3))
        unique_bigrams = list(set(bigrams))
        unique_trigrams = list(set(trigrams))

        class_counts = Counter(all_classes)
        class_report = "\\n".join([f"- {cls}: {count}" for cls, count in class_counts.items()])

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        word_freq = Counter([t for t in all_tokens if t.isalnum() and t not in stop_words])
        common_words = word_freq.most_common(10)
        words, counts = zip(*common_words) if common_words else ([], [])
        
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts, color='teal')
        plt.title("Top 10 most frequent words (without stopwords)")
        bar_file = f"stats_bar_{timestamp}.png"
        plt.savefig(bar_file)
        plt.close()

        lengths = [len(t) for t in all_tokens if t.isalnum()]
        plt.figure(figsize=(10, 5))
        plt.hist(lengths, bins=range(1, max(lengths) + 2 if lengths else 2), align='left', rwidth=0.8, color='orange')
        plt.title("Token Length Histogram in Database")
        hist_file = f"stats_hist_{timestamp}.png"
        plt.savefig(hist_file)
        plt.close()

        wc = WordCloud(width=800, height=400, background_color='white').generate(full_blob)
        wc_file = f"stats_wc_{timestamp}.png"
        wc.to_file(wc_file)

        report = (
            f"📊 **DATABASE STATISTICS**\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔹 **Unique tokens**: {len(unique_tokens)}\n"
            f"🔹 **Unique bigrams**: {len(unique_bigrams)}\n"
            f"🔹 **Unique trigrams**: {len(unique_trigrams)}\n\n"
            f"📈 **Class Distribution**:\n{class_report}\n\n"
            f"Sample unique tokens: `{', '.join(unique_tokens[:20])}`"
        )

        media = [
            InputMediaPhoto(open(bar_file, 'rb'), caption=report, parse_mode="Markdown"),
            InputMediaPhoto(open(hist_file, 'rb')),
            InputMediaPhoto(open(wc_file, 'rb'))
        ]
        
        await update.message.reply_media_group(media=media)
        
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        await update.message.reply_text(f"❌ Error: Data problems. {str(e)}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error during statistics generation: {str(e)}")


application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler('start', start))
application.add_handler(CommandHandler('task', task))
application.add_handler(CommandHandler('full_pipeline', full_pipeline))
application.add_handler(CommandHandler('classifier', classifier))
application.add_handler(CommandHandler('stats', stats))
application.run_polling()

