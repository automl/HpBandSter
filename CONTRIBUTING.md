## Contributing to HpBandSter

You are interested in developing a new feature or have found a bug? 
Awesome, feel welcome and read this guideline in order to find out how to best report your ideas so that we can include
them as quickly as possible.  

### Report security issues

You must never report security related issues, vulnerabilities or bugs including sensitive information to the bug tracker,
 or elsewhere in public. Instead sensitive bugs must be sent by email to one of the maintainers.

### New Features

We are always happy to read about your ideas on how to improve HpBandSter.
If you find yourself wishing for a feature that doesn't exist in HpBandster,
you are probably not alone. There are bound to be others out there with similar needs.
Open an issue on our [issues list on GitHub](https://github.com/automl/HpBandSter/issues),
 and describe 
- the feature you would like to see
- why you need it and
- how it should work.

If you already know how to implement, we love pull requests. 
Please see the [Pull request](#pull-requests) section, to read further details on pull requests.


### <a name="report-bugs"></a> Report Bugs

Report issues at <https://github.com/automl/HpBandSter/issues>

Before you report a bug, please make sure that:

1. Your bug hasn't already been reported in our [issue tracker](https://github.com/automl/HpBandSter/issues).
2. You are using the latest HpBandSter version.

If you found a bug, please provide us the following information:

- Your operating system name and version
- Any information about your setup that could be helpful to resolve the bug
- A simple example that reproduces that issue would be amazing. But if you can't provide an example, 
just note your observations as much detail as you can.
- Feel free, to add a screenshot showing the issue, if it helps.

If the issue needs an urgent fix, please mark it with the label "urgent".
Then either fix it or mark as "help wanted".

### Work on own features

To work on own features, first you need to create a fork of the original repository. 
A good tutorial on how to do this is in the Github Guide: [Fork a repo](https://help.github.com/articles/fork-a-repo/).

You could install the forked repository via:

<pre>
<code>git clone git@github.com:automl/HpBandSter.git
cd HpBandSter
python3 setup.py develop --user 
</code></pre>

### <a name="pull-requests"></a> Pull requests

If you have not worked with pull requests, you can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github).
Or read more in the official github documentation <https://help.github.com/articles/about-pull-requests/>

You know how to fix a bug or implement your own feature, follow this small guide:

- Check the issue tracker if someone has already reported the same idea or found the same bug. 
  (Note: If you only want to make some smaller changes, opening a new issue is less important, as the changes can be 
  discussed in the pull request.)
- Create a pull request, after you have implemented your changes in your fork and make a pull request.
  Using a separate branch for the fix is recommend.
- Pull request should include tests.
- We are using the Google Style Python Docstrings. Take a look at this [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). 
- The code should follow the PEP8 coding convention.
- We try to react as fast as possible to your pull request, but if you haven't received a feedback from us after some days
  feel free to leave a comment on the pull request. 
 