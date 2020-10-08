import os
import json
import datetime

import numpy as np


def generate_html(models):
    bootstrap = """
    <head>    
        <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">

        <style>
            td {
              padding: 5px;
            }

            th {
              padding: 5px;
            }

            h3 {
              margin-top: 0px;
            }
            .row {
              padding: 0.2em 0;
            }
            
            .row-striped:nth-of-type(odd){
              background-color: #efefef;
            }
            
            .row-striped:nth-of-type(even){
              background-color: #ffffff;
            }
        </style>
    </head>
    <body>
    <div class="row">
        <div class="col-md-11 col-md-offset-1">
            <h1>Trained models</h1>
        </div>
    </div>
    __models_html__
    </body>
    """

    base_model_html = """
    <div class="row row-striped">
        <div class="col-md-2 col-md-offset-1">
            <div class="row">
                <h3>__model_name__</h3>
            </div>
            <div class="row">
                <h4>__parsed_date__</h4>
            </div>
        </div>

        <div class="col-md-4">
            <div class="row">
                <h3>Training</h3>
            </div>
            <div class="row">
                <div class="col-md-3">
                    <p>Start date</p>
                </div>
                <div class="col-md-3">
                    <p>End date</p>
                </div>
                <div class="col-md-3">
                    <p>Epochs</p>
                </div>
                <div class="col-md-3">
                    <p>Training time</p>
                </div>
            </div>
            <div class="row">
                <div class="col-md-3">
                    __start_date__
                </div>
                <div class="col-md-3">
                    __end_date__
                </div>
                <div class="col-md-3">
                    __epochs__
                </div>
                <div class="col-md-3">
                    __training_time__
                </div>
            </div>
        </div>

        <div class="col-md-5">
            <div class="row">
                <h3>Metrics</h3>
            </div>
            <div class="row">
                <div class="col-md-3">
                    <p>Ticker</p>
                </div>
                <div class="col-md-3">
                    <p>1 day abs error</p>
                </div>
                <div class="col-md-3">
                    <p>5 day avg abs error</p>
                </div>
                <div class="col-md-3">
                    <p>14 day avg abs error</p>
                </div>
            </div>
            __metric_rows__
        </div>
    </div>
    """

    base_metric_row = """
    <div class="row">
        <div class="col-md-3">
            __ticker_name__
        </div>
        <div class="col-md-3">
            __metric1__
        </div>
        <div class="col-md-3">
            __metric2__
        </div>
        <div class="col-md-3">
            __metric3__
        </div>
    </div>
    """

    models_html = ""

    for model in models:
        model_html = base_model_html
        model_html = model_html.replace("__model_name__", model, 1)
        model_datetime = datetime.datetime.utcfromtimestamp(int(model)/1000).strftime('%Y-%m-%d %H:%M:%S')
        model_html = model_html.replace("__parsed_date__", model_datetime, 1)

        if type(models[model]['train_log']) == str:
            model_html = model_html.replace("__start_date__", "error", 1)
            model_html = model_html.replace("__end_date__", "error", 1)
            model_html = model_html.replace("__epochs__", "error", 1)
            model_html = model_html.replace("__training_time__", "error", 1)
            model_html = model_html.replace("__epochs__", "error", 1)
        else:
            model_html = model_html.replace("__start_date__", models[model]['train_log']['start_date'], 1)
            model_html = model_html.replace("__end_date__", models[model]['train_log']['end_date'], 1)
            model_html = model_html.replace("__training_time__", str(round(models[model]['train_log']['total_time'], 1))+"s", 1)

            if 'epochs' in models[model]['train_log']:
                model_html = model_html.replace("__epochs__", str(models[model]['train_log']['epochs']), 1)
            else:
                model_html = model_html.replace("__epochs__", "unknown", 1)

            metrics = models[model]['train_log']['metrics']

            # errors by 1 day, 5 days and 14 days (averaged)
            metric_rows = ""
            for ticker_name in metrics:
                metric_row = base_metric_row

                metric_row = metric_row.replace("__ticker_name__", ticker_name)

                err = metrics[ticker_name]['err']
                metric1 = np.abs(err[0])
                metric2 = np.abs(err[:5]).mean()
                metric3 = np.abs(err).mean()

                metric_row = metric_row.replace("__metric1__", f"{metric1:.2f}")
                metric_row = metric_row.replace("__metric2__", f"{metric2:.2f}")
                metric_row = metric_row.replace("__metric3__", f"{metric3:.2f}")

                metric_rows += metric_row

            model_html = model_html.replace("__metric_rows__", metric_rows)

        models_html += model_html

    bootstrap = bootstrap.replace("__models_html__", models_html)

    report_fn = "models_report.html"
    with open(report_fn, 'w') as f:
        f.write(bootstrap)
    
    print(f"created html report at: {report_fn}")


def create_html_report():
    src_dir = "./output"

    models = {}

    for model_name in sorted(os.listdir(src_dir)):
        if model_name.startswith("."):
            continue

        model_dict = {}

        model_dir = os.path.join(src_dir, model_name)

        with open(os.path.join(model_dir, "train.log")) as f:
            try:
                model_dict['train_log'] = json.loads(f.read())
            except json.JSONDecodeError:
                model_dict['train_log'] = "Error: couldn't load training log"

        models[model_name] = model_dict
        
    generate_html(models)


if __name__ == "__main__":
    create_html_report()
