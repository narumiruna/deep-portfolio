import click
import mlconfig


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('-c', '--config-file', default='configs/default.yaml')
def train(config_file):
    cfg = mlconfig.load(config_file)
    trainer = cfg.trainer(cfg, classmethod='from_config')
    trainer.fit()


if __name__ == '__main__':
    cli()
