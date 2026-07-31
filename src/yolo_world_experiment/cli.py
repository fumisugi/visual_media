from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visual-media-matplotlib")

import torch

from .coco import (
    build_final_holdout_manifest,
    build_holdout_manifest,
    build_prompt_prototype_holdout_manifest,
    build_same_model_holdout_manifest,
    build_subset_manifest,
)
from .config import load_config
from .experiment import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO-World failure-case experiment on COCO 2017 val"
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to the experiment YAML configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-gpu")
    subparsers.add_parser("prepare")
    subparsers.add_parser("prepare-holdout")
    subparsers.add_parser("prepare-final-holdout")
    subparsers.add_parser("prepare-same-model-holdout")
    subparsers.add_parser("prepare-prompt-prototype-holdout")
    subparsers.add_parser("pilot")
    subparsers.add_parser("full")
    subparsers.add_parser("prompt-study")
    subparsers.add_parser("improvement-dev")
    subparsers.add_parser("improvement-holdout")
    subparsers.add_parser("support-improvement-dev")
    subparsers.add_parser("final-holdout")
    subparsers.add_parser("model-scale-dev")
    subparsers.add_parser("model-scale-final")
    subparsers.add_parser("model-scale-visualize")
    subparsers.add_parser("same-model-blur-screen")
    subparsers.add_parser("same-model-blur-dev")
    subparsers.add_parser("same-model-blur-final")
    subparsers.add_parser("same-model-blur-visualize")
    subparsers.add_parser("prompt-prototype-screen")
    subparsers.add_parser("prompt-prototype-dev")
    subparsers.add_parser("prompt-prototype-final")
    subparsers.add_parser("prompt-prototype-visualize")
    subparsers.add_parser("extended-metrics")
    subparsers.add_parser("visualize")
    subparsers.add_parser("report")
    subparsers.add_parser("artifact")
    subparsers.add_parser("latex-report")
    subparsers.add_parser("validate")
    subparsers.add_parser("all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "check-gpu":
        payload = {
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
        print(json.dumps(payload, indent=2))
        if not payload["cuda_available"]:
            raise SystemExit(1)
        return

    if args.command == "prepare":
        manifest = build_subset_manifest(config)
        print(json.dumps(manifest["selection"], indent=2))
        return

    if args.command == "prepare-holdout":
        manifest = build_holdout_manifest(config)
        print(json.dumps(manifest["selection"], indent=2))
        return

    if args.command == "prepare-final-holdout":
        manifest = build_final_holdout_manifest(config)
        print(json.dumps(manifest["selection"], indent=2))
        return

    if args.command == "prepare-same-model-holdout":
        manifest = build_same_model_holdout_manifest(config)
        print(json.dumps(manifest["selection"], indent=2))
        return

    if args.command == "prepare-prompt-prototype-holdout":
        manifest = build_prompt_prototype_holdout_manifest(config)
        print(json.dumps(manifest["selection"], indent=2))
        return

    if args.command == "pilot":
        decision = ExperimentRunner(config).run_pilot()
        print(json.dumps(decision, indent=2))
        return

    if args.command == "full":
        result = ExperimentRunner(config).run_full()
        print(json.dumps(result["selected_parameters"], indent=2))
        return

    if args.command == "prompt-study":
        from .prompt_study import PromptStudy

        print(json.dumps(PromptStudy(config).run(), indent=2))
        return

    if args.command == "improvement-dev":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(config).run_development(), indent=2
            )
        )
        return

    if args.command == "improvement-holdout":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(config).run_holdout(), indent=2
            )
        )
        return

    if args.command == "support-improvement-dev":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(config).run_support_development(),
                indent=2,
            )
        )
        return

    if args.command == "final-holdout":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(config).run_final_holdout(),
                indent=2,
            )
        )
        return

    if args.command == "model-scale-dev":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(
                    config
                ).run_model_scale_development(),
                indent=2,
            )
        )
        return

    if args.command == "model-scale-final":
        from .clear_improvement import ClearImprovementStudy

        print(
            json.dumps(
                ClearImprovementStudy(
                    config
                ).run_model_scale_final_holdout(),
                indent=2,
            )
        )
        return

    if args.command == "model-scale-visualize":
        from .visualize import create_model_scale_outputs

        print("\n".join(str(path) for path in create_model_scale_outputs(config)))
        return

    if args.command == "same-model-blur-screen":
        from .same_model_improvement import SameModelImprovementStudy

        print(
            json.dumps(
                SameModelImprovementStudy(config).run_screening(),
                indent=2,
            )
        )
        return

    if args.command == "same-model-blur-dev":
        from .same_model_improvement import SameModelImprovementStudy

        print(
            json.dumps(
                SameModelImprovementStudy(config).run_development(),
                indent=2,
            )
        )
        return

    if args.command == "same-model-blur-final":
        from .same_model_improvement import SameModelImprovementStudy

        print(
            json.dumps(
                SameModelImprovementStudy(config).run_final_holdout(),
                indent=2,
            )
        )
        return

    if args.command == "same-model-blur-visualize":
        from .visualize import create_same_model_outputs

        print(
            "\n".join(
                str(path) for path in create_same_model_outputs(config)
            )
        )
        return

    if args.command == "prompt-prototype-screen":
        from .prompt_prototype_improvement import (
            PromptPrototypeImprovementStudy,
        )

        print(
            json.dumps(
                PromptPrototypeImprovementStudy(config).run_screening(),
                indent=2,
            )
        )
        return

    if args.command == "prompt-prototype-dev":
        from .prompt_prototype_improvement import (
            PromptPrototypeImprovementStudy,
        )

        print(
            json.dumps(
                PromptPrototypeImprovementStudy(config).run_development(),
                indent=2,
            )
        )
        return

    if args.command == "prompt-prototype-final":
        from .prompt_prototype_improvement import (
            PromptPrototypeImprovementStudy,
        )

        print(
            json.dumps(
                PromptPrototypeImprovementStudy(
                    config
                ).run_final_holdout(),
                indent=2,
            )
        )
        return

    if args.command == "prompt-prototype-visualize":
        from .visualize import create_prompt_prototype_outputs

        print(
            "\n".join(
                str(path)
                for path in create_prompt_prototype_outputs(config)
            )
        )
        return

    if args.command == "extended-metrics":
        from .extended_evaluation import run_extended_evaluation

        paths = run_extended_evaluation(config)
        print("\n".join(str(path) for path in paths.values()))
        return

    if args.command == "visualize":
        from .visualize import create_metric_charts, create_representative_images

        chart_paths = create_metric_charts(config)
        example_paths = create_representative_images(config)
        print("\n".join(str(path) for path in chart_paths + example_paths))
        return

    if args.command == "report":
        from .report import create_report_draft

        print(create_report_draft(config))
        return

    if args.command == "artifact":
        from .artifact import create_artifact

        print(create_artifact(config))
        return

    if args.command == "latex-report":
        from .latex_report import create_latex_report

        print(create_latex_report(config))
        return

    if args.command == "validate":
        from .validate import validate_results

        print(validate_results(config))
        return

    if args.command == "all":
        from .report import create_report_draft
        from .visualize import create_metric_charts, create_representative_images

        build_subset_manifest(config)
        runner = ExperimentRunner(config)
        decision = runner.run_pilot()
        if not decision["continue_yoloworld"]:
            print(json.dumps(decision, indent=2))
            raise SystemExit(2)
        runner.run_full()
        from .prompt_study import PromptStudy
        from .extended_evaluation import run_extended_evaluation

        PromptStudy(config).run()
        run_extended_evaluation(config)
        create_metric_charts(config)
        create_representative_images(config)
        print(create_report_draft(config))
        from .artifact import create_artifact

        print(create_artifact(config))
        from .validate import validate_results

        print(validate_results(config))
        from .latex_report import create_latex_report

        print(create_latex_report(config))


if __name__ == "__main__":
    main()
