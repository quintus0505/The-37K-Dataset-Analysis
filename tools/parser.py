from abc import ABC, abstractmethod
from data_loading import clean_participants_data, get_logdata_df, get_test_section_df


class Parse(ABC):
    def __init__(self):
        self.participants_dataframe = None
        self.logdata_dataframe = None
        self.test_sections_dataframe = None

        self.word_count = 0
        self.modified_word_count = 0
        self.auto_corrected_word_count = 0

        self.test_section_count = 0

    def load_participants(self):
        self.participants_dataframe = clean_participants_data()

    def load_logdata(self, full_log_data):
        self.logdata_dataframe = get_logdata_df(full_log_data)

    def load_test_sections(self):
        self.test_sections_dataframe = get_test_section_df()

    def compute_wmr(self, full_log_data):
        """
        Compute Word Modified Ratio (WMR): the ratio of words being modified during typing or after committed
        :param full_log_data: use the full log data or not
        :return:
        """
        self.load_participants()
        self.load_logdata(full_log_data)
        self.load_test_sections()

        # get those logdata with test sections id belonging to the selected participants
        participant_ids = self.participants_dataframe['PARTICIPANT_ID'].values

        # remove those rows where PARTICIPANT_ID is not in the selected participants
        self.test_sections_dataframe = self.test_sections_dataframe[
            self.test_sections_dataframe['PARTICIPANT_ID'].isin(participant_ids)]

        # remove those rows where test sections id is not in the selected test sections
        self.logdata_dataframe = self.logdata_dataframe[
            self.logdata_dataframe['TEST_SECTION_ID'].isin(self.test_sections_dataframe['TEST_SECTION_ID'])]

        # calculation based on the test sections, since the test section id are sorted, we can iterate through the
        # dataframe and calculate the word modified ratio

        # get the data with the same test section id
        test_section_ids = self.logdata_dataframe['TEST_SECTION_ID'].unique()

        for test_section_id in test_section_ids:
            try:
                # if test_section_id == 3098:
                #     print("test_section_id: ", test_section_id)
                # get the data with the same test section id
                test_section_df = self.logdata_dataframe[self.logdata_dataframe['TEST_SECTION_ID'] == test_section_id]

                # committed sentence from the last row of the test section
                committed_sentence = test_section_df.iloc[-1]['INPUT']

                # word modified flag with false value as a list of words in the committed sentence
                word_modified_flag = [False] * len(committed_sentence.split())
                if len(committed_sentence.split()) < 4:
                    continue
                for index, row in test_section_df.iterrows():
                    # if index == 20760:
                    #     print("test_section_id: ", test_section_id)

                    if row['INPUT'] != row['INPUT'] or row['DATA'] != row['DATA'] or row['INPUT'] == ' ' or row['DATA'] == ' ':
                        continue
                    if row['INPUT'] != committed_sentence[:len(row['INPUT'])]:
                        # compare each character of the input sentence with the committed sentence
                        committed_words = committed_sentence.split()
                        input_words = row['INPUT'].split()
                        typed_word_count = len(input_words)
                        last_word_index = 0
                        if typed_word_count > 1:
                            for i in range(min(typed_word_count, len(committed_words)) - 1):
                                if input_words[i] != committed_words[i]:
                                    if input_words[i] in committed_words:
                                        continue
                                    else:
                                        word_modified_flag[i] = True
                                last_word_index = i + 1

                        if len(input_words[last_word_index]) > len(committed_words[last_word_index]):
                            word_modified_flag[last_word_index] = True
                        else:
                            for j in range(len(input_words[last_word_index])):
                                if input_words[last_word_index][j] != committed_words[last_word_index][j]:
                                    word_modified_flag[last_word_index] = True

                self.test_section_count += 1
                test_section_word_count = len(committed_sentence.split())
                test_section_modified_word_count = sum(word_modified_flag)

                self.word_count += test_section_word_count
                self.modified_word_count += test_section_modified_word_count
                if test_section_modified_word_count / test_section_word_count >= 0.8:
                    print("test_section_id: ", test_section_id)
                # if self.test_section_count % 1 == 0:
                #     print("Word Modified Ratio (WMR): ", test_section_modified_word_count / test_section_word_count)

            except:
                pass

        print("Word Modified Ratio (WMR): ", self.modified_word_count / self.word_count)


if __name__ == "__main__":
    parser = Parse()
    parser.compute_wmr(full_log_data=True)
