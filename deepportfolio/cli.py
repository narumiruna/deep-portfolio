import click


@click.group()
def cli():
    pass


@cli.command(name='train')
def train():
    print('train model')


if __name__ == '__main__':
    cli()
