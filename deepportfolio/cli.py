import click
import mlconfig
import mlflow


@click.group()
def cli():
    pass


@cli.command(name='train')
@click.option('-c', '--config-file', default='configs/default.yaml')
def train(config_file):
    cfg = mlconfig.load(config_file)
    if hasattr(cfg, 'log_params'):
        mlflow.log_params(cfg.log_params)

    trainer = cfg.trainer(cfg, classmethod='from_config')
    trainer.fit()


if __name__ == '__main__':
    cli()
