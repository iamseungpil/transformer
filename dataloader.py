import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs

from torchtext.datasets import TranslationDataset

class Multi30k(TranslationDataset):
    """The small-dataset WMT 2016 multimodal task, also known as Flickr30k"""

    urls = ['https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
            'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/'
            'Multi30k/mmt16_task1_test.tar.gz']
    name = 'multi30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Multi30k, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)
    
class WMT14(TranslationDataset):
    """The WMT 2014 English-German dataset, as preprocessed by Google Brain.

    Though this download contains test sets from 2015 and 2016, the train set
    differs slightly from WMT 2015 and 2016 and significantly from WMT 2017."""

    urls = [('https://docs.google.com/uc?export=download&id=1zqVTc0sm5J-3_8OVaRrAM2X36CrRMqiS&confirm=t', 'data.zip')]
    name = 'wmt14'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train',
               validation='newstest2013',
               test='newstest2014', **kwargs):
        """Create dataset objects for splits of the WMT 2014 dataset.

        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.en', '.de') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default:
                'train.tok.clean.bpe.32000'.
            validation: The prefix of the validation data. Default:
                'newstest2013.tok.bpe.32000'.
            test: The prefix of the test data. Default:
                'newstest2014.tok.bpe.32000'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(WMT14, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)