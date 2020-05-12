import smtplib
from urllib import urlopen
from datetime import date
from bs4 import BeautifulSoup
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def scraper():
    link = 'https://en.wikipedia.org/wiki/Special:Random'
    page = urlopen(link)
    soup = BeautifulSoup(page, "html.parser")
    content = soup.find('div', id='bodyContent')
    if len(content) < 5:
        return scraper()
    else:
        return content


text = scraper()

fro = "arobert890@yahoo.com"
print "Enter the recepient's email address"
to = raw_input('>')
print "Scraping article..."
msg = MIMEMultipart()
msg['From'] = fro
msg['To'] = to
msg['Subject'] = 'Article of the day'

html = str(text)
part1 = MIMEText(html, 'html')
msg.attach(part1)
'''--------------------------------------------------------------------------------
--------------------------------------------------------------------------------'''
print "Connecting to SMTP server..."
server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
server.ehlo()
server.starttls()

print "Logging in..."
server.login('arobert890@yahoo.com', 'Sanramon6189')

server.sendmail(fro, to, str(msg))
print 'Done'

server.quit()
