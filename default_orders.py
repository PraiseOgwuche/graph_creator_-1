
known_orders = (
    ('Strongly disagree', 'Disagree',
     'Somewhat agree', 'Agree',
     'Strongly agree'),
    ('Less than 1 hour', '1-2 hours',
     '2-3 hours', '3-4 hours', 'More than 5 hours'),
    ('Extremely easy', 'Easy', 'Somewhat easy',
     'Somewhat difficult','Difficult',
     'Extremely difficult'),
    ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'),
    ('Not helpful at all', 'Not helpful', 'Somewhat helpful',
     'Helpful', 'Extremely helpful'),
    ('1 hour or less', '2 to 4 hours',
     '4 to 6 hours', '6 to 8 hours', '8 to 10 hours'),
    ('Extremely dissatisfied', 'Dissatisfied', 'Somewhat satisfied',
     'Satisfied', 'Extremely satisfied'),
    ('Not well', 'Slightly well',
     'Moderately well', 'Well',
     'Extremely well'),
    ('Not effective at all', 'Slightly effective', 'Somewhat effective',
     'Effective', 'Extremely effective'),
    ('Never', 'Rarely', 'Sometimes',
     'Often', 'Always'),
    ('Almost never', 'Rarely', 'Sometimes',
     'Frequently', 'Almost always'),
    ('Did not improve at all', 'Improved slightly', 'Improved neither too much nor too little',
     'Improved moderately', 'A lot'),
    ('Least helpful', 'Somewhat helpful', 'Moderately helpful',
     'Helpful', 'Extremely helpful'),
    ('30 minutes or less', '1-2 hours',
     '2-3 hours', '3-4 hours', '5 hours or more'),
    ('Very Low', 'Low', 'Medium', 'High', 'Very High'),
    ('Not effective at all', 'Slightly effective',
     'Moderately effective', 'Highly effective', 'Extremely effective'),
    ('Poorly', 'Slightly well',
     'Moderately well', 'Well',
     'Extremely well'),
    ('Extremely easy', 'Easy', 'Just right', 'Difficult', 'Extremely difficult'),
    ('Not satisfied at all', 'Slightly satisfied', 'Somewhat satisfied',
     'Satisfied', 'Extremely satisfied'),
    ('Not helpful at all', 'Slightly helpful', 'Somewhat helpful',
     'Helpful', 'Extremely helpful'),
    ('Not disappointed at all', 'Not very disappointed',
     'Somewhat disappointed', 'Disappointed',
     'Very disappointed'),
    ('Extremely Easy', 'Easy', 'Somewhat Difficult', 'Difficult',
     'Extremely difficult'),
    ('30 minutes or less', '30 to 60 minutes',
     '2-3 hours', '3-4 hours', '5 hours or more'),
    ('Network issues (e.g., my internet connection was slow or spotty)',
     'Account & login issues (e.g., I could not login to Forum)',
     'Audio/video issues (e.g., others could not hear or see me, I could not hear or see others)',
     'Other (please describe the problem)_________________'),
    ('Network issues (e.g., my internet connection was slow or spotty)',
     'Account & login issues (e.g., I could not login to Forum)',
     'Audio/video issues (e.g., others could not hear or see me, I could not hear or see others)',
     'Other (please describe the problem)'),
)


def check_if_order_is_known(order):
    for possible_order in known_orders:
        fit = True
        for term in order:
            if term not in possible_order:
                fit = False
                break
        if fit:
            return possible_order
