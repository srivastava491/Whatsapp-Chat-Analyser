from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path, encoding='utf-8') as file_obj:  # Ensure the file is read with utf-8 encoding
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Strip any extraneous whitespace

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# def get_requirements(file_name):
#     with open(file_name, encoding='utf-8') as file_obj:
#         requirements = file_obj.readlines()
#     return [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
setup(
    name='WhatsappChatAnalyser',
    version='0.0.1',
    author='Aarav Srivastava',
    author_email='aaravsrivastava491@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
