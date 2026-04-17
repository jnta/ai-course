from edgar import set_identity, Company
import sys

def main():
    set_identity("test@example.com")
    company = Company("AAPL")
    filing = company.get_filings(form="10-K").latest()
    filing_obj = filing.obj()
    print("Keys available:")
    try:
        print(filing_obj.items)
    except Exception as e:
        print(e)
    print(dir(filing_obj))
    print(type(filing_obj))
    print("part_i_item_1:", type(filing_obj['part_i_item_1']))

if __name__ == '__main__':
    main()
