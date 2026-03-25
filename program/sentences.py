import json
from os.path import exists


class Record:
    def __init__(self, text, classText):
        self.text = text
        self.classText = classText

    def json(self):
        return {
            "text": self.text,
            "classText": self.classText
        }

class Sentences:
    sentences = []

    @classmethod
    def init_sentences(cls):
        """Load sentences from JSON file with error handling"""
        try:
            if exists('./senteces.json'):
                with open('./senteces.json', 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    if not isinstance(data, list):
                        raise ValueError("JSON file must contain a list of records")
                    
                    for record in data:
                        if not isinstance(record, dict):
                            raise ValueError("Each record must be a JSON object")
                        if "text" not in record or "classText" not in record:
                            raise ValueError("Each record must contain 'text' and 'classText' fields")
                    
                    cls.sentences = data
            else:
                cls.sentences = []
                cls._save_sentences()
                
        except FileNotFoundError:
            raise FileNotFoundError("File 'senteces.json' not found")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"JSON syntax error in 'senteces.json': {str(e)}", 
                e.doc, 
                e.pos
            )
        except ValueError as e:
            raise ValueError(f"Data validation error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during JSON loading: {str(e)}")

    @classmethod
    def _save_sentences(cls):
        """Helper method for saving JSON data"""
        try:
            with open('./senteces.json', 'w', encoding='utf-8') as file:
                json.dump(cls.sentences, file, ensure_ascii=False, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write to 'senteces.json': {str(e)}")
        except Exception as e:
            raise Exception(f"Error during JSON saving: {str(e)}")

    @classmethod
    def add_record(cls, record: Record):
        """Add new record with error handling"""
        try:
            if not isinstance(record, Record):
                raise TypeError("Argument must be an instance of Record class")
            
            if not record.text or str(record.text).strip() == "":
                raise ValueError("'text' field cannot be empty")
            
            if not record.classText or str(record.classText).strip() == "":
                raise ValueError("'classText' field cannot be empty")
            
            Sentences.init_sentences()
            cls.sentences.append(record.json())
            cls._save_sentences()
            
        except TypeError as e:
            raise TypeError(f"Type error: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Validation error: {str(e)}")
        except IOError as e:
            raise IOError(f"I/O error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during record addition: {str(e)}")    


