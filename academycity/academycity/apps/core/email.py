from django.core.mail import send_mail


def email_message(semail, subject=None, body=None):
    email_from = 'noreply@drbaranes.com'
    if not subject:
        subject = 'Registration to GlobSim'
    if not body:
        body = 'Your registration has been accepted. you can login to http:\\www.academycity.org'
    return send_mail(subject, body, email_from, [semail], fail_silently=False)


def syn_email_message(subject, body, email_from, semail, fail_silently=False):
    return send_mail(subject, body, email_from, [semail], fail_silently=False)
