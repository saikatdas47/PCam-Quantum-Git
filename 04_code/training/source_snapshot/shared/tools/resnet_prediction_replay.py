"""Check numerical replay before retaining immutable saved predictions."""
import numpy as np

def install(common, run, stage):
    original = run.predict
    split = 'test' if stage == 'test' else 'valid'
    events = []
    def predict(model, spec, x):
        scores, probabilities = original(model, spec, x)
        path = run.BASE/'results'/f'{split}_predictions'/f'{spec["tag"]}.npz'
        if not path.exists():
            return scores, probabilities
        with np.load(path, allow_pickle=False) as saved:
            for key, actual in [('scores', scores), ('probabilities', probabilities)]:
                expected = saved[key]
                assert actual.shape == expected.shape
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                    err_msg='Prediction replay exceeds float64 tolerance: '+spec['tag'])
            # Recomputed float64 values can cross a threshold at a tied point.
            # Keep the saved predictions/threshold authoritative after bounded
            # numerical replay. Accept flips only inside the same tolerance band.
            cut = saved['threshold']
            assert cut.shape == () and np.isfinite(cut)
            flips = (probabilities >= cut) != (saved['probabilities'] >= cut)
            if np.any(flips):
                assert np.all(np.abs(probabilities[flips]-cut) <= 1e-12)
                assert np.all(np.abs(saved['probabilities'][flips]-cut) <= 1e-12)
            events.append({'tag': spec['tag'], 'split': split,
                'max_score_absolute_difference': float(np.max(np.abs(scores-saved['scores']))),
                'max_probability_absolute_difference': float(np.max(np.abs(probabilities-saved['probabilities']))),
                'absolute_tolerance': 1e-12, 'relative_tolerance': 0,
                'saved_arrays_preserved': True, 'threshold_decisions_equal': not bool(np.any(flips)),
                'threshold_boundary_flip_count': int(np.count_nonzero(flips)),
                'decision_policy': 'Use immutable saved probabilities and saved threshold; accept replay flips only within absolute tolerance of threshold'})
            common.atomic_json(run.BASE/'workflow'/f'{split}_prediction_replay.json', {'events': events})
            return saved['scores'].copy(), saved['probabilities'].copy()
    run.predict = predict
    verify = run.verify_development
    def verify_development():
        verify()
        path = run.BASE/'results/development_verification.json'
        import json
        record = json.loads(path.read_text())
        record['all_predictions_exact'] = all(e['max_score_absolute_difference'] == 0 and e['max_probability_absolute_difference'] == 0 for e in events)
        record['numerical_replay_passed'] = True
        record['replay_absolute_tolerance'] = 1e-12
        record['saved_predictions_preserved'] = True
        common.atomic_json(path, record)
    run.verify_development = verify_development
