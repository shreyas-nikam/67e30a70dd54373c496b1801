import sys

def test_imports():
    try:
        import streamlit
        import plotly
        import sklearn
        import art
        import secml
        import foolbox
        print('All libraries imported successfully.')
    except ImportError as e:
        print(f'ImportError: {e}')
        sys.exit(1)

def main():
    test_imports()
    print('Basic test completed successfully.')

if __name__ == '__main__':
    main()
