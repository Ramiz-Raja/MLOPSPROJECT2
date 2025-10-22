import requests, json

try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=10)
    print('HEALTH STATUS:', r.status_code)
    print(r.text)
except Exception as e:
    print('HEALTH ERROR:', repr(e))

try:
    payload = {'features': [5.1, 3.5, 1.4, 0.2]}
    r = requests.post('http://127.0.0.1:8000/predict', json=payload, timeout=10)
    print('\nPREDICT STATUS:', r.status_code)
    print(r.text)
except Exception as e:
    print('PREDICT ERROR:', repr(e))
