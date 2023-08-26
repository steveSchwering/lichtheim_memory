import Sentence
import Structure
import helperfunctions as hf

from pathlib import Path

class Producer():
    def __init__(self, structures_info_path):
        """
        Producer is essentially a wrapper around Structure to make accessing and working with sentence generation easier
        """
        self.structures = {structure.structure : structure for structure in Structure.read_structures(structures_info_path = structures_info_path)}
        self.structure_probs = {structure_str : structure_obj.structure_prob for structure_str, structure_obj in self.structures.items()}

    def generate_all_sentences(self):
        """
        """
        sentences = []
        for structure_name, structure in self.structures.items():
            sentences += structure.generate_all_sentences()

        return sentences

    def _sample_structure(self):
        """
        """
        return hf._choose_from_probability_dict(self.structure_probs)

    def generate_random_sentence(self, structure = None):
        """
        """
        if not structure:
            structure = self._sample_structure()
        
        return self.structures[structure].generate_random_sentence()

    def generate_random_sentences(self, structures = None, num_sentences = None):
        """
        Either structures or num_sentences must be specified

        structures should be a dictionary in the form of structure : num_sentences
        to denote how many sentences of each structure type should be generated

        num_sentences tells the Producer how many structures should be sampled before generating sentences
        """
        assert structures or num_sentences

        if not structures:
            # Create a counter to track how many sentences of each structure should be generated
            structures = {structure : 0 for structure in self.structures}
            for _ in range(num_sentences):
                structures[self._sample_structure()[0]] += 1

        # Generate a sentence for each chosen structure
        sentences = []
        for structure in structures:
            for _ in range(structures[structure]):
                sentences.append(self.generate_random_sentence(structure = structure))

        return sentences


if __name__ == '__main__':
    language = 'datives'

    structures_path = Path.cwd().joinpath(f'language/{language}/grammar/structures') 

    producer = Producer(structures_info_path = structures_path)
    all_sentences = producer.generate_all_sentences()
    all_sentences = Sentence.convert_sentence_dicts_to_objs(all_sentences)

    save_file = Path.cwd().joinpath(f'language/{language}/all_sentences.tsv')
    Sentence.save_sentences_info(all_sentences, save_file)

    print(all_sentences)