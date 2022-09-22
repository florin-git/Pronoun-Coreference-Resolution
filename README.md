# Homework #3 Coreference Resolution

This is the third homework of the NLP 2022 course at Sapienza University of Rome, taught by [**Roberto Navigli**](http://www.diag.uniroma1.it/~navigli/).

Check the [Report](https://github.com/florin-git/Pronoun-Coreference-Resolution/blob/main/report.pdf) and the Slide Presentation for more information.

The best model reaches an **accuracy score of 88.2%** on the secret test set of the course.
## Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project

### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.
For those who might be unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```bash
conda create -n nlp2022-hw3 python=3.9
conda activate nlp2022-hw3
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp2022-hw3
bash test.sh data/dev.tsv
```

Actually, you can replace *data/dev.tsv* to point to a different file, as far as the target file has the same format.

## Reproduce using checkpoints
You can download the checkpoints of the models I described in the paper from this [Google Drive link](https://drive.google.com/file/d/10flYiGKElB0IZsd2qrat_OR7c8QEk11H/view?usp=sharing).
